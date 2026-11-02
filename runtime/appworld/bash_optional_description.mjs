// Shared DSH execution compatibility, frozen with each new experiment.
// Model calls and training tokens stay verbatim; only the UI label defaults.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const upstream = new URL('../../runtime/dsh-v2/node_modules/@deepseek-ai/dsh-tool-bash/lib/index.js', import.meta.url);
export const UPSTREAM_SHA256 = '2ee3eebe0e22ba0bab055edafea0e8907b7ce63b6b8d803d0853a60375a04371';
export const ARGUMENT_POLICY = 'bash-description-default-v1';
export const DEFAULT_DESCRIPTION = 'Run the requested shell command';

if (createHash('sha256').update(readFileSync(upstream)).digest('hex') !== UPSTREAM_SHA256) {
  throw new Error('DSH bash source changed; review the optional-description adapter before running');
}
const implementation = await import(upstream.href);
export const { Config, inject, name } = implementation;

export function needsDescriptionDefault(args) {
  return args !== null && typeof args === 'object' && !Array.isArray(args)
    && (args.description === undefined
      || (typeof args.description === 'string' && args.description.trim() === ''));
}

export function executionArguments(args) {
  return needsDescriptionDefault(args) ? { ...args, description: DEFAULT_DESCRIPTION } : args;
}

export function wrapBashTool(tool) {
  if (tool.name !== 'bash' || tool.parameters?.properties?.description?.type !== 'string'
      || !tool.parameters.required?.includes('command')
      || !tool.parameters.required?.includes('description')) {
    throw new Error('Unexpected upstream bash schema; refusing argument adaptation');
  }
  const parameters = structuredClone(tool.parameters);
  parameters.required = parameters.required.filter(key => key !== 'description');
  parameters.properties.description.description = 'Optional UI description. If omitted or blank, the executor supplies a neutral label.';
  return {
    ...tool,
    parameters,
    async execute(args, exec) {
      // The upstream validator still checks every execution parameter, and
      // the same body retains sandbox policy, approval, cancellation and jobs.
      return tool.execute(executionArguments(args), exec);
    },
    ...(tool.presentCall ? { presentCall: args => tool.presentCall(executionArguments(args)) } : {}),
    ...(tool.presentResult ? { presentResult: (args, result) => tool.presentResult(executionArguments(args), result) } : {}),
    output: {
      ...tool.output,
      presentationMeta(args, value) {
        const original = tool.output.presentationMeta?.(executionArguments(args), value);
        return {
          ...(original === undefined ? {} : { upstream: original }),
          hapt_argument_defaults: {
            policy: ARGUMENT_POLICY,
            fields: needsDescriptionDefault(args) ? ['description'] : [],
          },
        };
      },
    },
  };
}

export function apply(ctx, config) {
  // Intercept only this plugin's registration, not the global registry or
  // shared node_modules. Keep the original scoped context and disposal owner.
  const registry = ctx.tools;
  const tools = new Proxy(registry, {
    get(target, key) {
      if (key === 'register') return tool => target.register(wrapBashTool(tool));
      return Reflect.get(target, key, target);
    },
  });
  const local = new Proxy(ctx, {
    get(target, key) {
      return key === 'tools' ? tools : Reflect.get(target, key, target);
    },
  });
  return implementation.apply(local, config);
}
