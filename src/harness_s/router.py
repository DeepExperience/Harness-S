"""Validation-based selection, with optional behavior profiles and LLM voting."""
import asyncio
from collections import Counter
import json
import logging
import math
import os
import random
import re
from .io import atomic_json, read_json, telemetry


class ModelClient:
    def __init__(self, config, directory):
        self.config, self.directory = config, directory
        self.limit = asyncio.Semaphore(config.get('concurrency',32))

    async def call(self, name, instruction, data):
        import httpx
        key = os.environ[self.config['api_key_env']]
        request = dict(model=self.config['model'], temperature=self.config.get('temperature',0),
            messages=[dict(role='system',content=instruction),dict(role='user',content=json.dumps(data,ensure_ascii=False))])
        telemetry(self.directory/f'{name}.request.json',request)
        async with self.limit:
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=600,trust_env=False) as client:
                        response = await client.post(self.config['base_url'].rstrip('/')+'/chat/completions',
                            headers={'Authorization':'Bearer '+key},json=request)
                        telemetry(self.directory/f'{name}.response-{attempt}.json',
                            dict(status=response.status_code,body=response.text))
                        response.raise_for_status()
                        raw = response.json()['choices'][0]['message']['content'].strip()
                        if raw.startswith('```'):
                            raw = raw.split('\n',1)[1].rsplit('```',1)[0]
                        return json.loads(raw)
                except (httpx.HTTPError, KeyError, ValueError):
                    if attempt==2:
                        raise
                    await asyncio.sleep(2**attempt)


async def build_profiles(observations, previous, client):
    profiles = {'h0':{'prefer':[],'avoid':[]}}
    async def one(hid, cases):
        differences = []
        for case in cases:
            if case.get('infrastructure_failure'):
                continue
            try:
                diff = await client.call(f'{hid}/differences/{case["task_id"]}',
                    'Return JSON {"differences": [...]}. Compare the sibling with H0 on this paired task. '
                    'Describe task conditions and concrete operations, including useful differences from a single case. '
                    'Use the actual trajectories. Do not infer broad capabilities from success alone.',case)
                differences.append(dict(task_id=case['task_id'],**diff))
            except Exception as exc:
                telemetry(client.directory/f'{hid}/unresolved/{case["task_id"]}.json',dict(reason=str(exc)))
        try:
            groups = await client.call(hid+'/groups',
                'Return JSON {"groups": [...]}. Group the supplied differences by similar task conditions '
                'and concrete operation sequences. Keep distinct singleton cases. Include source task IDs. '
                'Do not drop cases to meet a fixed count. Record reasons for unused cases.',differences)
            result = await client.call(hid+'/profile',
                'Return JSON {"prefer":[{"when":"...","behavior":"..."}],'
                '"avoid":[{"when":"...","behavior":"..."}]}. Write concise concrete routing advice '
                'relative to H0 using the supplied groups. Preserve all useful different conditions; '
                'merge duplicates only. Do not expose evidence IDs or history to the router.',groups)
            for side in ('prefer','avoid'):
                if not isinstance(result[side],list) or any(set(x)!={'when','behavior'} for x in result[side]):
                    raise ValueError('Invalid profile schema')
            profiles[hid] = result
        except Exception as exc:
            logging.warning('Keeping previous profile for %s: %s',hid,type(exc).__name__)
            profiles[hid] = previous.get(hid,{'prefer':[],'avoid':[]})
            telemetry(client.directory/f'{hid}/profile-fallback.json',dict(reason=str(exc)))
    await asyncio.gather(*(one(hid,cases) for hid,cases in observations.items() if hid!='h0'))
    return profiles


def aggregate(ballots, candidates, success_rates):
    scores = dict.fromkeys(candidates,0.)
    for ballot in ballots:
        ranked = ballot.get('ranked',[])
        if len(ranked)!=min(4,len(candidates)) or len(set(ranked))!=len(ranked) or any(h not in scores for h in ranked):
            raise ValueError('Each voter must rank four distinct eligible harnesses')
        for rank,hid in enumerate(ranked):
            scores[hid] += (4-rank)/4
    return sorted(candidates,key=lambda hid:(-scores[hid],-success_rates.get(hid,0),hid))


class RuleSelector:
    """Local TF-IDF cosine retrieval; rank by success on the same nearby tasks."""

    def __init__(self, config, stage, output):
        self.config, self.stage, self.output = config, stage, output
        self.validation = stage['validation']
        documents = [Counter(re.findall(r'\w+', x['instruction'].casefold())) for x in self.validation]
        frequency = Counter(token for document in documents for token in document)
        self.idf = {token: math.log((1+len(documents))/(1+count))+1 for token,count in frequency.items()}
        self.vectors = [self.vector(document) for document in documents]

    def vector(self, counts):
        weights = {token:(1+math.log(count))*self.idf[token] for token,count in counts.items() if token in self.idf}
        norm = math.sqrt(sum(x*x for x in weights.values()))
        return {token:weight/norm for token,weight in weights.items()} if norm else {}

    async def rank(self, task, candidates, inference=False):
        query = self.vector(Counter(re.findall(r'\w+', task['instruction'].casefold())))
        scores = [sum(weight*query.get(token,0) for token,weight in vector.items()) for vector in self.vectors]
        nearest = sorted((i for i,score in enumerate(scores) if score>0),
            key=lambda i:(-scores[i],self.validation[i]['task_id']))[:self.config.get('neighbors',5)]
        rates = self.stage['success_rates']
        local = {hid:sum(self.validation[i]['outcomes'][hid] for i in nearest)/len(nearest)
                 for hid in candidates} if nearest else {hid:rates.get(hid,0) for hid in candidates}
        ranking = sorted(candidates,key=lambda hid:(-local[hid],-rates.get(hid,0),hid))
        telemetry(self.output/task['task_id']/'ranking.json',dict(mode='rules',ranked=ranking,
            similar_validation_tasks=[dict(task_id=self.validation[i]['task_id'],similarity=scores[i]) for i in nearest],
            local_success_rates=local,success_rates=rates,fallback=not bool(nearest),
            stage=self.stage['step'],inference=inference))
        return ranking


def make_selector(config, stage, output):
    mode = config.get('mode','rules')
    if mode == 'rules':
        return RuleSelector(config,stage,output)
    if mode == 'llm':
        return Router(config,stage,output)
    raise ValueError(f'Unknown selection mode: {mode}')


class Router:
    def __init__(self, config, stage, output):
        self.config, self.stage = config, stage
        self.client = ModelClient(config,output)
        self.encoder = None
        self.validation_embeddings = None

    async def rank(self, task, candidates, inference=False):
        # Embeddings use validation instructions only, never Test outcomes.
        from sentence_transformers import SentenceTransformer
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.config['embedding_model'],device='cpu')
            self.validation_embeddings = self.encoder.encode(
                [x['instruction'] for x in self.stage['validation']],normalize_embeddings=True)
        query = self.encoder.encode([task['instruction']],normalize_embeddings=True)[0]
        scores = self.validation_embeddings @ query
        nearest = sorted(range(len(scores)),key=lambda i:-float(scores[i]))[:self.config.get('neighbors',5)]
        evidence = [self.stage['validation'][i] for i in nearest]
        async def vote(index):
            order = list(candidates)
            random.Random(f"{task['task_id']}:{self.stage['step']}:{index}").shuffle(order)
            backends = self.config.get('backends')
            client = self.client if not backends else ModelClient(
                {**self.config,**backends[index % len(backends)]},self.client.directory)
            return await client.call(f"{task['task_id']}/vote-{index}",
                'Return JSON {"ranked":[four distinct harness IDs]}. Rank task-appropriate harnesses '
                'using these behavior profiles, validation success rates and similar validation cases.',
                dict(task=task,candidates=order,profiles={h:self.stage.get('profiles',{}).get(h,{'prefer':[],'avoid':[]}) for h in order},
                     success_rates=self.stage['success_rates'],similar_validation_tasks=evidence))
        results = await asyncio.gather(*(vote(i) for i in range(self.config.get('voters',6))),return_exceptions=True)
        ballots = []
        for result in results:
            if isinstance(result,Exception):
                continue
            try:
                aggregate([result],candidates,self.stage['success_rates'])
                ballots.append(result)
            except ValueError:
                continue
        if not ballots:
            logging.warning('Router API unavailable; using latest validation ranking for %s',task['task_id'])
        ranking = aggregate(ballots,candidates,self.stage['success_rates'])
        telemetry(self.client.directory/task['task_id']/'ranking.json',dict(ranked=ranking,
            valid_ballots=len(ballots),fallback=not bool(ballots),stage=self.stage['step'],inference=inference))
        return ranking
