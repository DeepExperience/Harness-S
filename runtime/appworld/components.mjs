/** Safe declarative Context, Memory, Hook, and Tool-View runtime for AppWorld V5. */

export const name = 'hapt-appworld-v5-runtime'
export const inject = ['tools']

function stable(value) {
  if (Array.isArray(value)) return value.map(stable)
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])]))
  }
  return value
}

function contentText(content) {
  const output = []
  const visit = value => {
    if (typeof value === 'string') output.push(value)
    else if (Array.isArray(value)) value.forEach(visit)
    else if (value !== null && typeof value === 'object') {
      if (typeof value.text === 'string') output.push(value.text)
      else Object.values(value).forEach(visit)
    }
  }
  visit(content)
  return output.join('\n')
}

function pluginMessage(text, summary) {
  return {
    id: crypto.randomUUID(),
    role: 'user',
    content: [{ type: 'text', text }],
    source: { kind: 'plugin', plugin: name, form: 'notice', summary },
  }
}

function prepend(message, existing) {
  return [message, ...(existing ?? [])]
}

function initialState() {
  return {
    calls: 0,
    lastKey: '',
    repeatCount: 0,
    remainingSteps: undefined,
    lastErrorKind: 'none',
    schemaError: '',
    paginationCursor: '',
    authenticatedApps: new Set(),
    processedEntityKeys: new Set(),
    completionCalled: false,
    revision: 0,
  }
}

function classifyError(text, isError) {
  if (/\b401\b|invalid credentials|unauthori[sz]ed/i.test(text)) return 'authentication_401'
  if (/\b422\b|validation error|field required/i.test(text)) return 'schema_422'
  if (/traceback|execution failed|exception:/i.test(text)) return 'python_or_api_exception'
  if (isError) return 'tool_error'
  return 'none'
}

function updateState(state, exec, result) {
  const before = JSON.stringify(stable({
    remainingSteps: state.remainingSteps,
    lastErrorKind: state.lastErrorKind,
    schemaError: state.schemaError,
    paginationCursor: state.paginationCursor,
    authenticatedApps: [...state.authenticatedApps],
    processedEntityKeys: [...state.processedEntityKeys],
    completionCalled: state.completionCalled,
  }))
  const key = JSON.stringify([exec.name, stable(exec.arguments)])
  state.repeatCount = state.lastKey === key ? state.repeatCount + 1 : 1
  state.lastKey = key
  state.calls += 1
  const code = typeof exec.arguments?.code === 'string' ? exec.arguments.code : ''
  const text = contentText(result.content)
  const remaining = text.match(/["']?remaining_steps["']?\s*[:=]\s*(\d+)/i)
  if (remaining) state.remainingSteps = Number(remaining[1])
  state.lastErrorKind = classifyError(text, result.isError)
  if (state.lastErrorKind === 'schema_422') {
    state.schemaError = text.replace(/\s+/g, ' ').slice(0, 600)
  } else if (!result.isError) {
    state.schemaError = ''
  }
  const cursor = text.match(/["']?(?:next_cursor|next_page_token|cursor)["']?\s*[:=]\s*["']?([^,"'\s}\]]+)/i)
  state.paginationCursor = cursor?.[1] && !/^(?:none|null|false)$/i.test(cursor[1]) ? cursor[1] : ''
  if (/apis\.supervisor\.complete_task\s*\(/.test(code)) state.completionCalled = true
  if (!result.isError) {
    for (const match of code.matchAll(/apis\.([a-zA-Z0-9_]+)\.login\s*\(/g)) state.authenticatedApps.add(match[1])
    for (const match of text.matchAll(/["'](?:id|uuid|email|name)["']\s*:\s*["']([^"']{1,100})["']/g)) {
      if (state.processedEntityKeys.size < 50) state.processedEntityKeys.add(match[1])
    }
  }
  const after = JSON.stringify(stable({
    remainingSteps: state.remainingSteps,
    lastErrorKind: state.lastErrorKind,
    schemaError: state.schemaError,
    paginationCursor: state.paginationCursor,
    authenticatedApps: [...state.authenticatedApps],
    processedEntityKeys: [...state.processedEntityKeys],
    completionCalled: state.completionCalled,
  }))
  if (before !== after) state.revision += 1
  return before !== after
}

function extractorValue(state, extractor) {
  if (extractor === 'remaining_steps') return state.remainingSteps ?? 'unknown'
  if (extractor === 'last_error_kind') return state.lastErrorKind
  if (extractor === 'schema_error') return state.schemaError || 'none'
  if (extractor === 'pagination_cursor') return state.paginationCursor || 'none'
  if (extractor === 'authenticated_apps') return [...state.authenticatedApps].sort()
  if (extractor === 'processed_entity_keys') return [...state.processedEntityKeys].sort()
  if (extractor === 'completion_called') return state.completionCalled
  return 'unsupported'
}

function memorySnapshot(state, memory, maxChars) {
  if (!memory.enabled) return ''
  const rows = memory.fields.map(field => `${field.render_label}=${JSON.stringify(extractorValue(state, field.extractor))}`)
  const text = `<hapt_episode_memory>\n${rows.join('\n')}\n</hapt_episode_memory>`
  return text.length <= maxChars ? text : `${text.slice(0, maxChars)}…`
}

function conditionMatches(hook, state, exec, resultText = '') {
  const params = hook.parameters ?? {}
  const code = typeof exec.arguments?.code === 'string' ? exec.arguments.code : ''
  if (hook.condition === 'always') return true
  if (hook.condition === 'duplicate_call_gte') {
    const key = JSON.stringify([exec.name, stable(exec.arguments)])
    const count = state.lastKey === key ? state.repeatCount + 1 : 1
    return count >= Number(params.threshold ?? 3)
  }
  if (hook.condition === 'forbidden_apis_import') return /(^|\n)\s*(?:from\s+apis\b|import\s+apis\b)|apis\.apps\b/.test(code)
  if (hook.condition === 'remaining_steps_lte') return state.remainingSteps !== undefined && state.remainingSteps <= Number(params.threshold ?? 2)
  if (hook.condition === 'last_error_is') return state.lastErrorKind === String(params.kind ?? '')
  if (hook.condition === 'pagination_present') return Boolean(state.paginationCursor) || /next[_ ]?(?:cursor|page|token)|has[_ ]?more/i.test(resultText)
  if (hook.condition === 'memory_missing') return extractorValue(state, String(params.extractor ?? '')) === 'none'
  return false
}

function helperAnalysis(code) {
  const findings = []
  if (/(^|\n)\s*(?:from\s+apis\b|import\s+apis\b)/.test(code)) findings.push('Do not import apis; the persistent REPL already provides it.')
  if (/apis\.apps\b/.test(code)) findings.push('Use apis.<app>.<api> directly; apis.apps is unsupported.')
  if (!/\bapis\./.test(code)) findings.push('No AppWorld API call was detected in this draft.')
  if (/apis\.supervisor\.complete_task\s*\(/.test(code)) findings.push('Completion call detected; verify all requested side effects first.')
  return { valid: findings.length === 0, findings }
}

export function apply(ctx, config) {
  const context = config?.context ?? { state_injection: { mode: 'never', max_chars: 1200 } }
  const memory = config?.memory ?? { enabled: false, fields: [] }
  const hooks = Array.isArray(config?.hooks) ? config.hooks : []
  const toolView = config?.tool_view ?? { description_suffix: '', examples: [], error_normalization: false, helper_tool: false }
  const states = new WeakMap()
  const stateFor = agent => {
    let state = states.get(agent)
    if (!state) {
      state = initialState()
      states.set(agent, state)
    }
    return state
  }

  if (toolView.description_suffix || toolView.examples?.length) {
    ctx.on('system-prompt/assemble', async (_assembly, _context, next) => {
      const downstream = await next()
      const addition = [toolView.description_suffix, ...(toolView.examples ?? []).map(value => `Example: ${value}`)]
        .filter(Boolean).join('\n')
      return {
        ...downstream,
        tools: downstream.tools.map(tool => tool.name === 'mcp__environment__execute'
          ? { ...tool, description: `${tool.description ?? ''}\n\n${addition}`.trim() }
          : tool),
      }
    })
  }

  if (toolView.helper_tool) {
    ctx.tools.register({
      name: 'harness_validate_execute',
      description: 'Perform a side-effect-free static check of a proposed AppWorld execute code string. This never calls the environment.',
      parameters: { code: { type: 'string', required: true } },
      output: {
        schema: {
          type: 'object',
          properties: {
            valid: { type: 'boolean' },
            findings: { type: 'array', items: { type: 'string' } },
          },
          required: ['valid', 'findings'],
        },
        render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
      },
      async execute(args) { return helperAnalysis(String(args.code ?? '')) },
    })
  }

  ctx.on('tools/pre-execute', async (exec, next) => {
    if (!exec.agent || exec.name !== 'mcp__environment__execute') return next()
    const state = stateFor(exec.agent)
    const code = typeof exec.arguments?.code === 'string' ? exec.arguments.code : ''
    const isCompletion = /apis\.supervisor\.complete_task\s*\(/.test(code)
    const relevant = hooks.filter(hook => hook.event === 'before_tool' || (isCompletion && hook.event === 'before_complete'))
    for (const hook of relevant) {
      if (!conditionMatches(hook, state, exec)) continue
      // A pre-execution warning deliberately denies this draft so the Policy can revise it without an environment side effect.
      return { kind: 'deny', reason: hook.message }
    }
    return next()
  })

  ctx.on('tools/post-execute', async (exec, result, next) => {
    if (!exec.agent || exec.name !== 'mcp__environment__execute') return next()
    const state = stateFor(exec.agent)
    const text = contentText(result.content)
    const changed = updateState(state, exec, result)
    const notices = []
    for (const hook of hooks.filter(value => value.event === 'after_tool')) {
      if (!conditionMatches(hook, state, exec, text)) continue
      if (hook.action === 'warn' || hook.action === 'normalize_error') notices.push(hook.message)
    }
    if (toolView.error_normalization && state.lastErrorKind !== 'none') {
      notices.push(`Normalized AppWorld error: kind=${state.lastErrorKind}; inspect the documented API schema before retrying.`)
    }
    const mode = context?.state_injection?.mode ?? 'never'
    const shouldInject = mode === 'always' || (mode === 'on_change' && changed) || (mode === 'on_error' && state.lastErrorKind !== 'none')
    if (shouldInject) {
      const snapshot = memorySnapshot(state, memory, Number(context?.state_injection?.max_chars ?? 1200))
      if (snapshot) notices.push(snapshot)
    }
    const downstream = await next()
    if (notices.length === 0) return downstream
    const message = pluginMessage(notices.join('\n\n'), 'AppWorld V5 Harness update')
    if (downstream.kind === 'block') {
      return { kind: 'block', feedback: downstream.feedback, additionalContexts: prepend(message, downstream.additionalContexts) }
    }
    return { ...downstream, additionalContexts: prepend(message, downstream.additionalContexts) }
  })
}
