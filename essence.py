
#!/usr/bin/env python3
# Essence workspace CLI -- standalone, no external Essence dependency.
# Commands: probe  install  pull <model>  chat  tui  skills  mcp  vault
from __future__ import annotations
import sys, os, json, logging as _logging, subprocess, shutil
from pathlib import Path

# Module-level logger — used in _run() coroutine error paths
_log = _logging.getLogger("essence")
_WS = Path(__file__).resolve().parent
os.environ.setdefault('ESSENCE_WORKSPACE', str(_WS))
try:
    import tomllib
except ImportError:
    try: import tomli as tomllib  # type: ignore
    except ImportError: tomllib = None  # type: ignore
def _cfg():
    p = _WS / 'config.toml'
    if tomllib and p.exists():
        with open(p,'rb') as f: return tomllib.load(f)
    return {}
def _model(): return os.environ.get('ESSENCE_MODEL', _cfg().get('inference',{}).get('model','qwen3:4b'))
def _ollama(): return os.environ.get('OLLAMA_HOST', os.environ.get('ESSENCE_OLLAMA_HOST','http://localhost:11434'))
def _b(s): return f'\033[1m{s}\033[0m'
def _c(s): return f'\033[96m{s}\033[0m'
def _g(s): return f'\033[92m{s}\033[0m'
def _r(s): return f'\033[91m{s}\033[0m'
def _d(s): return f'\033[2m{s}\033[0m'
BANNER='\n\u2554'+'\u2550'*60+'\u2557\n\u2551  Essence  --  Essence — Your Private AI'+' '*17+'\u2551\n\u255a'+'\u2550'*60+'\u255d\n'
def cmd_probe():
    import platform
    cfg=_cfg()
    print(BANNER)
    print(f'  Workspace   {_WS}')
    print(f'  OS/Arch     {platform.system()} / {platform.machine()}')
    print(f'  Python      {sys.version.split()[0]}')
    try:
        import psutil  # type: ignore
        print(f'  RAM         {round(psutil.virtual_memory().total/1024**3,1)} GB')
    except ImportError: pass
    print(f'  Backend     {cfg.get("inference",{}).get("backend","ollama")}')
    print(f'  Model       {_model()}')
    try:
        import httpx
        r=httpx.get(f'{_ollama()}/api/tags',timeout=3)
        tags=[m['name'] for m in r.json().get('models',[])]
        print(f'  Ollama      {_g("online")}  ({len(tags)} models)')
        ok=_g('ready') if _model() in tags else _r(f'not pulled -- run: python essence.py pull {_model()}')
        print(f'  {_model():<16} {ok}')
    except Exception: print(f'  Ollama      {_r("offline")}')
    print()
def cmd_install():
    pkgs=['httpx>=0.27', 'psutil>=5.9', 'python-json-logger>=2.0', 'openai>=1.30', 'tiktoken>=0.7', 'textual>=0.63', 'huggingface_hub>=0.23', 'pandas>=2.0', 'numpy>=1.24', 'scikit-learn>=1.4', 'scipy>=1.12', 'matplotlib>=3.8', 'joblib>=1.3', 'pypdf>=4.0', 'python-docx>=1.0', 'beautifulsoup4>=4.12', 'vaderSentiment>=3.3', 'cryptography>=42.0', 'keyring>=25.0']
    print(BANNER)
    print('  Installing workspace dependencies ...\n')
    subprocess.run([sys.executable,'-m','pip','install','--upgrade']+pkgs,check=True,timeout=600)
    if not shutil.which('ollama'):
        print('\n  WARNING: Ollama not found -- install from https://ollama.com/download')
    print(f'\n  {_g("Done.")}')
def cmd_pull(tag):
    if not shutil.which('ollama'): print(_r('Ollama not installed. https://ollama.com')); sys.exit(1)
    subprocess.run(['ollama','pull',tag],check=True,timeout=600)
def cmd_chat():
    try: import httpx
    except ImportError: print(_r('httpx not installed. Run: python essence.py install')); sys.exit(1)
    model,base=_model(),_ollama()
    hist=[]
    soul=_WS/'SOUL.md'
    if soul.exists(): hist.append({'role':'system','content':soul.read_text(encoding='utf-8')})
    print(BANNER)
    print(f'  Model  {_c(model)}   ({base})   /exit to quit\n')
    while True:
        try: msg=input(_b('You: ')).strip()
        except (EOFError,KeyboardInterrupt): print(); break
        if not msg: continue
        if msg.lower() in ('/exit','/quit'): break
        hist.append({'role':'user','content':msg})
        print(_b('AI:'),end=' ',flush=True)
        buf=''
        try:
            with httpx.Client(timeout=120) as c:
                with c.stream('POST',f'{base}/api/chat',
                              json={'model':model,'messages':hist,'stream':True}) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line: continue
                        try: d=json.loads(line)
                        except Exception: continue
                        t=d.get('message',{}).get('content','')
                        print(t,end='',flush=True); buf+=t
                        if d.get('done'): break
        except Exception as e: print(_r(f'\n  Error: {e}')); hist.pop(); continue
        print(); hist.append({'role':'assistant','content':buf})
def cmd_skills(args):
    """Manage skills from the CLI."""
    sys.path.insert(0,str(_WS))
    skills_dir = _WS / 'memory' / 'skills'
    try:
        from server.skillstore import SkillStore
    except ImportError:
        print(_r('skillstore not found. Run from the Essence directory.')); sys.exit(1)
    ss = SkillStore(skills_dir)
    sub = args[0] if args else 'list'
    if sub == 'list':
        skills = ss.list_all()
        # Merge in-process SkillAgent skills not already in SkillStore
        try:
            from server.skill_orchestrator import BUILTIN_SKILLS
            store_ids = {s['id'] for s in skills}
            for sd in BUILTIN_SKILLS:
                if sd.id not in store_ids:
                    skills.append({
                        'id': sd.id, 'label': sd.name, 'category': sd.category,
                        'enabled': sd.enabled, 'builtin': True, '_source': 'in-process',
                        'description': sd.description,
                    })
        except Exception:
            pass
        print(f'\n  Essence Skills\n  {"="*60}')
        print(f'  {"ID":<20} {"LABEL":<30} {"CAT":<14} {"EN":<5} {"SOURCE"}')
        print('  ' + '-'*83)
        for s in skills:
            en = _g('yes') if s.get('enabled') else _r('no ')
            src = s.get('_source', 'builtin' if s.get('builtin') else 'json_store')
            print(f'  {s["id"]:<20} {s["label"]:<30} {s["category"]:<14} {en}  {src}')
        print(f'\n  {len(skills)} skills · {sum(1 for s in skills if s.get("enabled"))} enabled')
    elif sub == 'add':
        # Parse --name --desc --category --prompt from args
        def _arg(flag):
            try: idx=args.index(flag); return args[idx+1]
            except (ValueError,IndexError): return ''
        label = _arg('--name') or _arg('--label')
        if not label: print(_r('Usage: python essence.py skills add --name "My Skill" [--category tool] [--desc "..."] [--prompt "..."]')); sys.exit(1)
        data = {'label':label,'category':_arg('--category') or 'custom','description':_arg('--desc'),'system_prompt':_arg('--prompt')}
        s = ss.create(data)
        print(f'  {_g("Created:")} {s["id"]} — {s["label"]}')
    elif sub == 'from-md':
        path = args[1] if len(args)>1 else ''
        if not path or not Path(path).exists(): print(_r(f'File not found: {path}')); sys.exit(1)
        data = SkillStore.from_markdown(Path(path).read_text(encoding='utf-8'))
        s = ss.create(data)
        print(f'  {_g("Imported from MD:")} {s["id"]} — {s["label"]}')
    elif sub == 'from-json':
        path = args[1] if len(args)>1 else ''
        if not path or not Path(path).exists(): print(_r(f'File not found: {path}')); sys.exit(1)
        import json as _json
        data = _json.loads(Path(path).read_text(encoding='utf-8'))
        items = data if isinstance(data,list) else [data]
        for item in items:
            try: s=ss.create(item); print(f'  {_g("Imported:")} {s["id"]}')
            except Exception as e: print(f'  {_r("Skip:")} {item.get("id","?")} — {e}')
    elif sub in ('enable','disable'):
        sid = args[1] if len(args)>1 else ''
        if not sid: print(_r(f'Usage: python essence.py skills {sub} <skill_id>')); sys.exit(1)
        try: s=ss.patch(sid,{'enabled': sub=='enable'}); print(f'  {s["id"]}: {_g("enabled") if s["enabled"] else _r("disabled")}')
        except KeyError: print(_r(f'Skill not found: {sid}'))
    elif sub == 'remove':
        sid = args[1] if len(args)>1 else ''
        if not sid: print(_r('Usage: python essence.py skills remove <skill_id>')); sys.exit(1)
        try:
            ok = ss.delete(sid)
            print(f'  {_g("Removed:")} {sid}' if ok else _r(f'Not found: {sid}'))
        except ValueError as e: print(_r(str(e)))
    elif sub == 'show':
        sid = args[1] if len(args)>1 else ''
        s = ss.get(sid)
        if not s: print(_r(f'Skill not found: {sid}')); sys.exit(1)
        import json as _json
        print(_json.dumps(s, indent=2))
    elif sub == 'search':
        import asyncio as _aio
        query = ' '.join(args[1:]) if len(args) > 1 else ''
        if not query: print(_r('Usage: python essence.py skills search <query>')); sys.exit(1)
        try:
            from server.skill_marketplace import get_marketplace
            mp = get_marketplace()
            results = _aio.run(mp.search(query))
            if not results:
                print(f'  No marketplace skills matching "{query}"')
            else:
                print(f'\n  Marketplace results for "{query}" ({len(results)} found)\n  {"="*60}')
                print(f'  {"ID":<28} {"NAME":<28} {"VERSION":<8}  {"CATEGORY"}')
                print('  ' + '-'*72)
                for r in results:
                    print(f'  {r["id"]:<28} {r["name"]:<28} {r.get("version","?"):<8}  {r.get("category","?")}')
                print(f'\n  Install: python essence.py skills install <id>')
            print()
        except Exception as e:
            print(_r(f'  Marketplace search failed: {e}'))
    elif sub == 'install':
        import asyncio as _aio
        sid = args[1] if len(args) > 1 else ''
        if not sid: print(_r('Usage: python essence.py skills install <id>')); sys.exit(1)
        force = '--force' in args
        try:
            from server.skill_marketplace import install_skill
            result = _aio.run(install_skill(sid, skills_dir, force=force))
            if result['ok']:
                print(f'  {_g("Installed:")} {sid}')
                print(f'  Path:    {result.get("path","?")}')
                print(f'  Status:  {result.get("message","")}')
            else:
                print(_r(f'  Install failed: {result["message"]}'))
        except Exception as e:
            print(_r(f'  Install error: {e}'))
    else:
        print(f'  Unknown skills sub-command: {sub}')
        print('  Sub-commands: list  add  from-md  from-json  enable  disable  remove  show  search  install')

def cmd_mcp(args):
    """Manage MCP servers from the CLI."""
    sys.path.insert(0,str(_WS))
    mcp_path = _WS / 'memory' / 'ledger' / 'mcp_servers.json'
    try:
        from server.mcpstore import MCPStore
    except ImportError:
        print(_r('mcpstore not found. Run from the Essence directory.')); sys.exit(1)
    ms = MCPStore(mcp_path)
    sub = args[0] if args else 'list'
    if sub == 'list':
        servers = ms.list_all()
        print(f'\n  Essence MCP Servers\n  {"="*60}')
        print(f'  {"ID":<18} {"LABEL":<28} {"TRANSPORT":<8} {"EN":<5} {"TOOLS"}')
        print('  ' + '-'*72)
        for s in servers:
            en = _g('yes') if s.get('enabled') else _r('no ')
            nt = len(s.get('tools',[]))
            print(f'  {s["id"]:<18} {s["label"]:<28} {s["transport"]:<8} {en}  {nt} tool(s)')
    elif sub == 'add':
        def _arg(flag):
            try: idx=args.index(flag); return args[idx+1]
            except (ValueError,IndexError): return ''
        label     = _arg('--name') or _arg('--label')
        command   = _arg('--command')
        url       = _arg('--url')
        transport = _arg('--transport') or ('http' if url else 'stdio')
        if not label: print(_r('Usage: python essence.py mcp add --name "MyServer" [--command "npx ..."] [--url "http://..."]')); sys.exit(1)
        cmd_list  = command.split() if command else []
        s = ms.register({'label':label,'transport':transport,'command':cmd_list,'url':url})
        print(f'  {_g("Registered:")} {s["id"]} — {s["label"]}')
    elif sub == 'remove':
        sid = args[1] if len(args)>1 else ''
        ok = ms.delete(sid)
        print(f'  {_g("Removed:")} {sid}' if ok else _r(f'Not found: {sid}'))
    elif sub in ('enable','disable'):
        sid = args[1] if len(args)>1 else ''
        if not sid: print(_r(f'Usage: python essence.py mcp {sub} <server_id>')); sys.exit(1)
        try: s=ms.patch(sid,{'enabled':sub=='enable'}); print(f'  {s["id"]}: {_g("enabled") if s["enabled"] else _r("disabled")}')
        except KeyError: print(_r(f'MCP server not found: {sid}'))
    else:
        print(f'  Unknown mcp sub-command: {sub}')
        print('  Sub-commands: list  add  remove  enable  disable')

def cmd_tui():
    """Bootstrap the kernel stack then launch the TUI in the same process."""
    import asyncio, sys
    from pathlib import Path

    # Add server/ to path so imports work
    srv = _WS / 'server'
    if str(srv) not in sys.path:
        sys.path.insert(0, str(_WS))

    data_dir = _WS / 'data'
    data_dir.mkdir(exist_ok=True)

    # Install redacting log formatter on any file handlers so secrets never
    # appear in essence.log / server.log even if the user pastes a key into chat.
    try:
        import logging as _logging
        from server.redact import RedactingFormatter as _RF
        _fmt = _RF('%(asctime)s %(levelname)s %(name)s: %(message)s')
        for _h in _logging.root.handlers:
            if isinstance(_h, _logging.FileHandler):
                _h.setFormatter(_fmt)
    except Exception:
        pass

    try:
        from server.event_log       import EventLog
        from server.event_bus       import init_event_bus
        from server.gravity_memory  import GravityMemory
        from server.identity_engine import IdentityEngine
        from server.governance      import GovernanceEnforcer, TrustLedger, CIEScorer
        from server.skill_agent     import SkillRegistry
        from server.action_engine   import PatternRegistry, ActionEngine
        from server.kernel          import init_kernel

        # 1. Governance primitives
        trust   = TrustLedger()
        cie     = CIEScorer()

        # 2. Event log + bus (bus needs log for persistence)
        ev_log  = EventLog(data_dir / 'event_log.db')
        bus     = init_event_bus(ev_log)

        # 2a. Audit logger — records governance events, model switches, config changes
        from server.audit_logger import init_audit_logger as _init_audit
        _init_audit(data_dir / 'audit_log.db')

        # 2b. Offline cache — session resilience and message queue
        from server.offline_cache import init_offline_cache as _init_offline
        _init_offline(cache_dir=data_dir / 'offline_cache',
                      backend_url=_get_ollama())

        # 2c. Service registry — learned external APIs (re-registers dynamic tools on startup)
        from server.service_registry import init_service_registry as _init_svc_reg
        from server.service_ingestor import init_ingestor as _init_ingestor
        _init_svc_reg(data_dir / 'service_registry.db')
        _init_ingestor(ollama_url=_get_ollama(), model=_get_model())

        # 3. Memory + identity (wire bus for protected write paths)
        memory   = GravityMemory(data_dir / 'gravity_memory.db', event_bus=bus)
        identity = IdentityEngine(_WS, event_bus=bus)

        # 4. Governance enforcer
        gov = GovernanceEnforcer(bus, trust, cie)

        # 5. Skill registry — register built-in skills and sync SkillStore state
        skills = SkillRegistry()
        _register_builtin_skills(skills)
        _sync_skill_states_from_store(skills, _WS / 'memory' / 'skills')

        # 6. Action engine (pattern-based proactive triggers)
        from server.action_engine import PatternRegistry, ActionEngine, _DEFAULT_DB as _pat_db
        pat_reg = PatternRegistry(_pat_db)
        action_eng = ActionEngine(pat_reg, bus)

        # 7. Episodic memory (L2) — session_id from config or fresh uuid
        from server.episodic_memory import init_episodic_memory
        import uuid as _uuid
        session_id = _cfg().get('session', {}).get('id', _uuid.uuid4().hex[:12])
        episodic = init_episodic_memory(session_id=session_id,
                                        db_path=data_dir / 'episodic.db')

        # 7b. Context compressor — intelligent conversation window management
        from server.context_compressor import init_compressor as _init_compressor
        _ctx_tokens = int(_cfg().get('inference', {}).get('context_tokens', 32_768))
        _init_compressor(ollama_url=_get_ollama(), model=_get_model(),
                         context_tokens=_ctx_tokens)

        # 8. Model selection — build router now (sync), select inside _run (async)
        ollama_url     = _get_ollama()
        _config_model  = _get_model()        # value from config / env
        _config_backend = _get_backend()     # e.g. "ollama" | "groq" | "openai" …

        # Detect performance tier from available RAM
        _tier = 'standard'
        try:
            import psutil as _ps
            _ram = _ps.virtual_memory().total / 1024**3
            _tier = ('lite'    if _ram <  8 else
                     'standard' if _ram < 16 else
                     'mem-opt'  if _ram < 32 else 'gpu-acc')
        except ImportError:
            pass

        from server.model_router import ModelRouter, HFInferenceBackend

        # Load saved provider keys from config.toml so API keys set via
        # `/provider set groq api_key ...` are available after a restart.
        _saved_provider_cfg: dict = {}
        try:
            _full_cfg = _cfg()
            for _pkey, _pval in _full_cfg.items():
                if _pkey.startswith("provider.") and isinstance(_pval, dict):
                    _pid = _pkey[len("provider."):]
                    _saved_provider_cfg[_pid] = dict(_pval)
            # Also load env-var keys that match known providers
            import os as _os
            for _pid in ("groq", "openai", "anthropic", "mistral", "deepseek",
                         "together", "gemini", "openrouter"):
                _env_key = _pid.upper() + "_API_KEY"
                _env_val = _os.environ.get(_env_key) or _os.environ.get(f"ESSENCE_{_pid.upper()}_KEY")
                if _env_val:
                    _saved_provider_cfg.setdefault(_pid, {})["api_key"] = _env_val
        except Exception:
            pass

        _router = ModelRouter(_WS, ollama_url=ollama_url, perf_tier=_tier,
                              provider_cfg=_saved_provider_cfg)

        async def _run():
            # ── Dynamic model selection (async) ─────────────────────
            # If config explicitly names a non-Ollama backend, honour it directly
            # and skip the local model probe so it doesn't override the choice.
            if _config_backend and _config_backend != 'ollama':
                _provider = _config_backend
                model     = _config_model or _router._default_model(_config_backend)
                print(f'[info] Using configured backend: {_provider}  model: {model}')
            else:
                available = await _router.get_ollama_models()
                if _config_model and _config_model in available:
                    _provider, model = 'ollama', _config_model
                else:
                    _provider, model = await _router.select_best_available(
                        task_type='chat',
                        hitl=len(available) > 1,
                        n_suggest=8,
                    )

            _hf_backend = None
            if _provider == 'hf_local':
                _model_path = _WS / 'models' / model
                _hf_backend = HFInferenceBackend(_model_path)
                if not _hf_backend.is_ready():
                    print('[warn] llama-cpp-python not installed; falling back to Ollama default')
                    _hf_backend = None
                    model       = _config_model or 'qwen3:4b'

            # ── Tool engine workspace init ────────────────────────── (P0 fix)
            from server.tools_engine import init_tools as _init_tools
            _init_tools(_WS)

            # ── Kernel init ─────────────────────────────────────────
            kernel = init_kernel(bus, memory, identity, gov, skills, ollama_url, model,
                                 episodic=episodic)
            kernel._provider   = _provider        # honour configured/selected provider
            # Propagate saved provider keys into kernel's internal router and dynamic router
            if _saved_provider_cfg:
                kernel._router._provider_cfg.update(_saved_provider_cfg)
                kernel._dynamic_router.update_available(_saved_provider_cfg)

            # Initialise model tier registry (PRIMARY / FALLBACK / EMERGENCY).
            # Loads [models.*] from config.toml so tiers persist across restarts.
            try:
                from server.model_tiers import init_tier_registry as _init_tiers
                _tier_reg = _init_tiers(_WS)
                kernel._tier_registry = _tier_reg
            except Exception as _mte:
                _log.debug("ModelTiers init skipped: %s", _mte)

            if _hf_backend is not None:
                kernel._hf_backend = _hf_backend  # fallback inference path

            # Snapshot config + tool definitions for offline resilience
            try:
                from server.offline_cache import get_offline_cache as _get_oc
                _oc = _get_oc()
                _oc.save_config({
                    'model': model, 'provider': _provider,
                    'ollama_url': ollama_url, 'tier': _tier,
                })
                _oc.save_tool_definitions(skills.catalog_for_planner())
            except Exception as _oce:
                _log.warning('Offline cache config snapshot failed: %s', _oce)

            await skills.load_all(bus)
            kernel.start()
            action_eng.start()

            # Plugin SDK — wire dispatch_hook into bus for each supported hook
            try:
                from server.plugin_sdk import PluginStore, dispatch_hook
                _plugin_store = PluginStore(_WS / 'memory' / 'plugins')
                _plugin_hooks = ['user.request', 'agent.message', 'skill.result',
                                 'heartbeat.tick', 'trigger.fired', 'cron.tick']
                for _hook in _plugin_hooks:
                    async def _make_plugin_handler(h=_hook, ps=_plugin_store):
                        async def _handler(env):
                            await dispatch_hook(ps, h, env.data)
                        return _handler
                    bus.subscribe(_hook, await _make_plugin_handler())
            except Exception as _pe:
                _log.warning('Plugin SDK wiring failed: %s', _pe)

            # ProactiveAgent — background autonomous check-ins
            _proactive_agent = None
            try:
                from server.proactive_agent import ProactiveAgent
                _proactive_agent = ProactiveAgent(
                    workspace=_WS,
                    ollama_url=ollama_url,
                    model=model,
                    tick_interval=900,
                )
                _proactive_agent.start()
            except Exception as _pae:
                _log.warning('ProactiveAgent start failed: %s', _pae)

            # Trigger Network — filesystem/RSS/IMAP event watchers
            _trigger_net = None
            try:
                from server.trigger_network import init_trigger_network
                _trigger_net = init_trigger_network(_WS, bus)
                _trigger_net.start()
            except Exception as _tne:
                _log.warning('TriggerNetwork start failed: %s', _tne)

            # Agent Mesh — peer discovery and task delegation
            _mesh = None
            try:
                import os as _os
                if _os.environ.get('ESSENCE_MESH_ENABLED', '').lower() in ('true', '1'):
                    from packages.essence_agents.essence_agents.mesh import init_mesh
                    _mesh = init_mesh(_WS, bus)
                    _mesh.start()
            except Exception as _me:
                _log.debug('AgentMesh start failed: %s', _me)

            # REST API — optional HTTP interface (set ESSENCE_REST_PORT to activate)
            _rest_task = None
            try:
                import os as _os
                _rest_port = int(_os.environ.get('ESSENCE_REST_PORT', '0'))
                if _rest_port:
                    from server.rest_api import start_rest_api
                    _rest_task = asyncio.create_task(
                        start_rest_api(bus, kernel, _WS, port=_rest_port),
                        name='essence-rest-api',
                    )
            except Exception as _rae:
                _log.warning('REST API start failed: %s', _rae)

            # Wake-word detector (optional, enabled by ESSENCE_WAKE_ENABLED=true)
            _wake = None
            try:
                import os as _os
                if _os.environ.get('ESSENCE_WAKE_ENABLED', '').lower() in ('true', '1'):
                    from server.wake_word import get_wake_detector
                    _wake = get_wake_detector(bus)
                    _wake.start()
            except Exception as _we:
                _log.debug('WakeWord start failed: %s', _we)

            # Knowledge Graph — init singleton so it persists across turns
            try:
                from server.knowledge_graph import get_knowledge_graph as _get_kg
                _get_kg()   # creates DB if not exists
            except Exception:
                pass

            # Memory sync — restore from S3 on startup (no-op when unconfigured)
            try:
                from server.memory_sync import restore as _ms_restore
                await _ms_restore(_WS)
            except Exception:
                pass

            # Run the TUI
            from tui_app import EssenceTUI
            app = EssenceTUI()
            await app.run_async()

            # ── Graceful teardown ──────────────────────────────────────
            if _proactive_agent:
                _proactive_agent.stop()
            if _trigger_net:
                _trigger_net.stop()
            if _mesh:
                _mesh.stop()
            if _wake:
                _wake.stop()
            if _rest_task and not _rest_task.done():
                _rest_task.cancel()
            action_eng.stop()
            await skills.unload_all()
            await kernel.shutdown()   # writes session summary to episodic DB

            # Memory sync — backup to S3 on clean shutdown
            try:
                from server.memory_sync import backup as _ms_backup
                await _ms_backup(_WS)
            except Exception:
                pass

        asyncio.run(_run())

    except ImportError as e:
        # Graceful fallback: kernel deps missing, run TUI standalone
        print(f'[warn] Kernel bootstrap failed ({e}) — running TUI without kernel')
        t = _WS / 'tui_app.py'
        if t.exists():
            import subprocess
            subprocess.run([sys.executable, str(t)], check=False)
        else:
            cmd_chat()


def _get_model() -> str:
    cfg = _cfg()
    import os
    return os.environ.get('ESSENCE_MODEL', cfg.get('inference', {}).get('model', 'qwen3:4b'))


def _get_backend() -> str:
    """Return the configured inference backend (provider id). Defaults to 'ollama'."""
    import os
    return os.environ.get('ESSENCE_BACKEND', _cfg().get('inference', {}).get('backend', 'ollama'))


def _get_ollama() -> str:
    import os
    return os.environ.get('OLLAMA_HOST', os.environ.get('ESSENCE_OLLAMA_HOST', 'http://localhost:11434'))


def _register_builtin_skills(registry) -> None:
    """Register built-in SkillAgent implementations."""
    try:
        from server.skills.research_skill  import ResearchSkill
        registry.register(ResearchSkill())
    except ImportError:
        pass
    try:
        from server.skills.memory_skill import MemorySkill
        registry.register(MemorySkill())
    except ImportError:
        pass
    try:
        from server.skills.summarise_skill import SummariseSkill
        registry.register(SummariseSkill())
    except ImportError:
        pass

def _sync_skill_states_from_store(registry, skills_dir) -> None:
    """Apply SkillStore enabled/disabled state to SkillRegistry in-process agents."""
    try:
        from server.skillstore import SkillStore
        store = SkillStore(skills_dir)
        for s in store.list_all():
            sid = s.get("id", "")
            agent = registry._agents.get(sid)
            if agent is not None:
                agent.manifest.enabled = bool(s.get("enabled", True))
    except Exception as exc:
        import logging
        logging.getLogger("essence").warning("SkillStore sync failed: %s", exc)


def cmd_version():
    cfg=_cfg()
    ver=cfg.get('version','1.0.0')
    print(f'\n  Essence  v{ver}')
    print(f'  Python   {sys.version.split()[0]}')
    print(f'  Workspace {_WS}')
    print()

def cmd_models(args):
    """List / pull / remove Ollama models and manage local HuggingFace GGUFs."""
    sub = args[0] if args else 'list'
    try:
        import httpx
    except ImportError:
        print(_r('httpx not installed. Run: python essence.py install')); sys.exit(1)

    if sub == 'list':
        try:
            r = httpx.get(f'{_ollama()}/api/tags', timeout=5)
            models = r.json().get('models', [])
            print(f'\n  Ollama Models  ({_ollama()})\n  {"="*60}')
            print(f'  {"NAME":<36} {"SIZE":>8}  {"MODIFIED"}')
            print('  ' + '-'*60)
            cur = _model()
            for m in models:
                name = m.get('name', '?')
                size = m.get('size', 0)
                mod  = m.get('modified_at', '')[:10]
                arrow = _c('◄') if name == cur else ' '
                print(f'  {arrow} {name:<34} {size//1024**3:>5} GB  {mod}')
            print(f'\n  {len(models)} model(s)  ·  current: {_c(cur)}')
        except Exception as e:
            print(_r(f'  Cannot reach Ollama: {e}'))

    elif sub == 'pull':
        tag = args[1] if len(args) > 1 else ''
        if not tag:
            print(_r('Usage: python essence.py models pull <name>')); sys.exit(1)
        cmd_pull(tag)

    elif sub == 'rm':
        tag = args[1] if len(args) > 1 else ''
        if not tag:
            print(_r('Usage: python essence.py models rm <name>')); sys.exit(1)
        subprocess.run(['ollama', 'rm', tag], check=True)

    elif sub == 'select':
        # Interactive best-model selection (same HITL flow used at TUI boot)
        import asyncio as _aio
        sys.path.insert(0, str(_WS))
        try:
            from server.model_router import ModelRouter
            router = ModelRouter(_WS, ollama_url=_get_ollama())
            provider, model_name = _aio.run(
                router.select_best_available(hitl=True, n_suggest=10)
            )
            print(f'\n  {_g("Selected:")} [{provider}] {model_name}')
            print(f'  To use this model, set in config.toml:')
            print(f'  model = "{model_name}"')
        except Exception as e:
            print(_r(f'  Error: {e}'))

    elif sub == 'hf-list':
        # List locally downloaded GGUF models
        sys.path.insert(0, str(_WS))
        from server.model_router import ModelRouter
        router = ModelRouter(_WS, ollama_url=_get_ollama())
        local  = router.scan_local_models()
        print(f'\n  Local GGUF Models  ({_WS / "models"})\n  {"="*60}')
        if not local:
            print('  No local GGUF models.')
            print('  Pull one with:  python essence.py models hf-pull')
            print()
            return
        print(f'  {"FILE":<44} {"SIZE":>8}')
        print('  ' + '-'*56)
        for m in local:
            print(f'  {m["file"]:<44} {m["size_gb"]:>6.2f} GB')
        print(f'\n  {len(local)} model(s)  ·  directory: {_WS / "models"}')

    elif sub == 'hf-pull':
        # Pull a GGUF from HuggingFace (suggested or explicit)
        import asyncio as _aio
        sys.path.insert(0, str(_WS))
        from server.model_router import ModelRouter

        # Resolve tier from RAM
        tier = 'standard'
        try:
            import psutil as _ps
            ram = _ps.virtual_memory().total / 1024**3
            tier = 'lite' if ram < 8 else 'standard' if ram < 16 else 'mem-opt' if ram < 32 else 'gpu-acc'
        except ImportError:
            pass

        router = ModelRouter(_WS, ollama_url=_get_ollama(), perf_tier=tier)
        hf_token = os.environ.get('ESSENCE_HF_TOKEN', '') or os.environ.get('HF_TOKEN', '')

        if len(args) >= 3:
            # explicit: models hf-pull <repo> <filename>
            repo, filename = args[1], args[2]
            result = _aio.run(router.pull_from_hf(repo, filename, hf_token=hf_token))
        else:
            print(f'  Auto-selecting best GGUF for tier {_c(tier)} …')
            result = _aio.run(router.suggest_and_pull_hf())

        if result:
            print(f'  {_g("Downloaded:")} {result}')
        else:
            print(_r('  Pull failed. Check logs or try: python essence.py models hf-pull <repo> <file>'))

    elif sub == 'hf-run':
        # Launch a local GGUF chat session via llama-cpp-python
        sys.path.insert(0, str(_WS))
        from server.model_router import HFInferenceBackend, ModelRouter
        router = ModelRouter(_WS, ollama_url=_get_ollama())
        local  = router.scan_local_models()
        if not local:
            print(_r('  No local GGUF models. Run: python essence.py models hf-pull'))
            sys.exit(1)

        # Pick model: explicit filename or first available
        filename = args[1] if len(args) > 1 else local[0]['file']
        model_path = _WS / 'models' / filename
        if not model_path.exists():
            print(_r(f'  Model file not found: {model_path}')); sys.exit(1)

        backend = HFInferenceBackend(model_path)
        if not backend.is_ready():
            print(_r('  llama-cpp-python not installed. Run: pip install llama-cpp-python'))
            sys.exit(1)

        print(BANNER)
        print(f'  Running local model: {_c(filename)}   /exit to quit\n')
        soul = _WS / 'SOUL.md'
        hist = []
        if soul.exists():
            hist.append({'role': 'system', 'content': soul.read_text(encoding='utf-8')})
        while True:
            try:
                msg = input(_b('You: ')).strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not msg: continue
            if msg.lower() in ('/exit', '/quit'): break
            hist.append({'role': 'user', 'content': msg})
            print(_b('AI:'), end=' ', flush=True)
            buf = ''
            try:
                for tok in backend.chat(hist, stream=True):
                    print(tok, end='', flush=True)
                    buf += tok
            except Exception as e:
                print(_r(f'\n  Error: {e}')); hist.pop(); continue
            print()
            hist.append({'role': 'assistant', 'content': buf})

    else:
        print(f'  Unknown sub-command: {sub}')
        print('  Ollama sub-commands: list  pull <name>  rm <name>  select')
        print('  HuggingFace sub-commands: hf-list  hf-pull [<repo> <file>]  hf-run [<file>]')
    print()

def cmd_logs(args):
    """Show recent log lines."""
    n=50
    for a in args:
        if a.startswith('--tail='):
            try: n=int(a.split('=')[1])
            except: pass
        elif a=='--tail' and args.index(a)+1<len(args):
            try: n=int(args[args.index(a)+1])
            except: pass
    candidates=[_WS/'server.log',_WS/'server_service.log',_WS/'essence.log']
    for p in candidates:
        if p.exists():
            lines=p.read_text(encoding='utf-8',errors='replace').splitlines()
            print(f'\n  {p.name}  (last {n} lines)\n  {"="*60}')
            for line in lines[-n:]:
                if 'ERROR' in line or 'error' in line: print(_r(f'  {line}'))
                elif 'WARN' in line: print(f'\033[93m  {line}\033[0m')
                else: print(f'  {line}')
            print()
            return
    print(_r('  No log file found.'))

def cmd_cost(args):
    """Show session cost + token summary with pricing-table estimates."""
    sys.path.insert(0, str(_WS))
    # Try real-time usage tracker first (populated when TUI / kernel is running)
    try:
        from server.usage_pricing import get_usage_tracker
        summary = get_usage_tracker().summary()
        print()
        for line in summary.splitlines():
            print(f'  {line}')
        print()
    except Exception:
        pass

    # Fall back to persisted cost_log.json for historical data
    cost_file = _WS / 'memory' / 'cost_log.json'
    if cost_file.exists():
        try:
            data = json.loads(cost_file.read_text())
            if data:
                total = sum(e.get('cost', 0) for e in data)
                ti    = sum(e.get('tok_in', 0)  for e in data)
                to    = sum(e.get('tok_out', 0) for e in data)
                print(f'  Historical (from cost_log.json)')
                print(f'  {"─"*48}')
                print(f'  Sessions      {len(data)}')
                print(f'  Tokens in     {ti:>12,}')
                print(f'  Tokens out    {to:>12,}')
                print(f'  Total tokens  {ti+to:>12,}')
                print(f'  Est. cost     {_c(f"${total:.6f}")}')
                recent = data[-5:]
                if recent:
                    print(f'\n  Last {len(recent)} sessions:')
                    print(f'  {"SESSION":<20} {"MODEL":<20} {"TOKENS":>8}  {"COST"}')
                    print('  ' + '─' * 58)
                    for e in reversed(recent):
                        t_total = e.get('tok_in', 0) + e.get('tok_out', 0)
                        print(f'  {e.get("session","?"):<20} {e.get("model","?"):<20} {t_total:>8,}  ${e.get("cost",0):.6f}')
                print()
        except Exception:
            pass

def cmd_snapshot(args):
    """Save or list session snapshots."""
    snaps_dir=_WS/'memory'/'snapshots'
    snaps_dir.mkdir(parents=True,exist_ok=True)
    sub=args[0] if args else 'list'
    if sub=='list':
        snaps=sorted(snaps_dir.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        if not snaps: print('  No snapshots saved.'); return
        print(f'\n  Snapshots ({len(snaps)})\n  {"="*60}')
        print(f'  {"ID":<32} {"DATE":<18} {"MSGS":>5}  {"COST"}')
        print('  '+'-'*62)
        for p in snaps[:20]:
            try:
                d=json.loads(p.read_text())
                from datetime import datetime
                ts=datetime.fromtimestamp(d.get('timestamp',0)).strftime('%Y-%m-%d %H:%M')
                msgs=len([m for m in d.get('messages',[]) if m.get('role')!='system'])
                print(f'  {d.get("id","?"):<32} {ts:<18} {msgs:>5}  ${d.get("cost",0):.4f}')
            except: print(f'  {p.stem}')
        print()
    elif sub=='save':
        import uuid,time
        sid=args[1] if len(args)>1 else uuid.uuid4().hex[:8]
        snap={'id':sid,'timestamp':time.time(),'model':_model(),'messages':[],'cost':0}
        path=snaps_dir/f'{sid}.json'
        path.write_text(json.dumps(snap,indent=2))
        print(f'  {_g("Saved:")} {sid}')
    elif sub=='rm':
        sid=args[1] if len(args)>1 else ''
        if not sid: print(_r('Usage: python essence.py snapshot rm <id>')); sys.exit(1)
        p=snaps_dir/f'{sid}.json'
        if p.exists(): p.unlink(); print(f'  {_g("Removed:")} {sid}')
        else: print(_r(f'  Not found: {sid}'))
    else:
        print(f'  Unknown sub-command: {sub}')
        print('  Sub-commands: list  save [id]  rm <id>')

def cmd_cron(args):
    """Manage cron jobs."""
    cron_file=_WS/'memory'/'cron_jobs.json'
    def _load():
        try: return json.loads(cron_file.read_text()) if cron_file.exists() else []
        except: return []
    def _save(j): cron_file.write_text(json.dumps(j,indent=2))
    sub=args[0] if args else 'list'
    if sub=='list':
        jobs=_load()
        print(f'\n  Cron Jobs ({len(jobs)})\n  {"="*60}')
        if not jobs: print('  No jobs. Use: python essence.py cron add <name> <schedule>'); print(); return
        print(f'  {"ID":<8} {"NAME":<18} {"SCHEDULE":<18} {"EN":<5} {"LAST STATUS"}')
        print('  '+'-'*60)
        for j in jobs:
            en=_g('yes') if j.get('enabled',True) else _r('no ')
            print(f'  {j.get("id","?"):<8} {j.get("name","?"):<18} {j.get("schedule",""):<18} {en}  {j.get("last_status","idle")}')
        print()
    elif sub=='add':
        import uuid
        name=args[1] if len(args)>1 else ''
        sched=args[2] if len(args)>2 else '*/30 * * * *'
        cmd_str=' '.join(args[3:]) if len(args)>3 else ''
        if not name: print(_r('Usage: python essence.py cron add <name> <schedule> [command]')); sys.exit(1)
        jobs=_load()
        job={'id':uuid.uuid4().hex[:6],'name':name,'schedule':sched,'command':cmd_str,
             'enabled':True,'last_run':'','last_status':'idle'}
        jobs.append(job); _save(jobs)
        print(f'  {_g("Added:")} {job["id"]} — {name} [{sched}]')
    elif sub in ('enable','disable'):
        name=args[1] if len(args)>1 else ''
        if not name: print(_r(f'Usage: python essence.py cron {sub} <name|id>')); sys.exit(1)
        jobs=_load()
        for j in jobs:
            if j.get('name')==name or j.get('id')==name:
                j['enabled']=sub=='enable'
        _save(jobs)
        print(f'  {_g(sub+"d:")} {name}')
    elif sub=='rm':
        name=args[1] if len(args)>1 else ''
        if not name: print(_r('Usage: python essence.py cron rm <name|id>')); sys.exit(1)
        jobs=[j for j in _load() if j.get('name')!=name and j.get('id')!=name]
        _save(jobs); print(f'  {_g("Removed:")} {name}')
    else:
        print(f'  Unknown sub-command: {sub}')
        print('  Sub-commands: list  add <name> <schedule>  enable  disable  rm')

def cmd_health():
    """Quick system health check."""
    import platform
    try: import httpx
    except ImportError: print(_r('httpx not installed. Run: python essence.py install')); sys.exit(1)
    print(f'\n  Essence Health Check\n  {"="*44}')
    # Python
    print(f'  Python      {_g(sys.version.split()[0])}')
    # Ollama
    try:
        import httpx as hx
        r=hx.get(f'{_ollama()}/api/tags',timeout=3)
        tags=[m['name'] for m in r.json().get('models',[])]
        print(f'  Ollama      {_g("online")}  {_ollama()}')
        cur=_model()
        pulled=_g('pulled') if cur in tags else _r('not pulled')
        print(f'  Model       {_c(cur)}  {pulled}')
    except Exception as e:
        print(f'  Ollama      {_r("offline")}  {e}')
    # Workspace
    print(f'  Workspace   {_g("ok")}  {_WS}')
    # Memory dir
    mem=_WS/'memory'
    print(f'  Memory dir  {_g("ok") if mem.exists() else _r("missing")}')
    # Config
    cfg_path=_WS/'config.toml'
    print(f'  config.toml {_g("ok") if cfg_path.exists() else _r("missing")}')
    # SOUL
    soul=_WS/'SOUL.md'
    print(f'  SOUL.md     {_g("ok") if soul.exists() else _r("missing")}')
    # psutil
    try:
        import psutil  # type: ignore
        mem_info=psutil.virtual_memory()
        print(f'  RAM         {round(mem_info.available/1024**3,1)}GB free / {round(mem_info.total/1024**3,1)}GB total')
    except ImportError:
        print(f'  psutil      {_r("not installed")}')
    print()

def cmd_config(args):
    """Show or set config values."""
    cfg_path=_WS/'config.toml'
    sub=args[0] if args else 'show'
    if sub=='show':
        if not cfg_path.exists(): print('  No config.toml found.'); return
        print(f'\n  config.toml\n  {"="*44}')
        print(cfg_path.read_text())
    elif sub=='get':
        key=args[1] if len(args)>1 else ''
        if not key: print(_r('Usage: python essence.py config get <key>')); sys.exit(1)
        cfg=_cfg()
        parts=key.split('.')
        val=cfg
        for p in parts:
            if isinstance(val,dict): val=val.get(p)
            else: val=None; break
        print(f'  {key} = {val}')
    elif sub=='set':
        print(_r('  config set not yet implemented. Edit config.toml directly.'))
    else:
        print(f'  Unknown sub-command: {sub}')
        print('  Sub-commands: show  get <key>')

def cmd_memory(args):
    """Browse or search memory entries."""
    mem_dir=_WS/'memory'
    sub=args[0] if args else 'list'
    if sub=='list':
        files=list(mem_dir.rglob('*.json'))
        print(f'\n  Memory Files ({len(files)})\n  {"="*60}')
        for f in sorted(files)[:30]:
            size=f.stat().st_size
            rel=f.relative_to(_WS)
            print(f'  {str(rel):<50} {size:>8} B')
        print()
    elif sub=='stats':
        total=sum(f.stat().st_size for f in mem_dir.rglob('*') if f.is_file())
        count=sum(1 for _ in mem_dir.rglob('*.json'))
        print(f'\n  Memory Stats')
        print(f'  JSON files  {count}')
        print(f'  Total size  {total//1024} KB')
        snaps=list((_WS/'memory'/'snapshots').glob('*.json')) if (_WS/'memory'/'snapshots').exists() else []
        print(f'  Snapshots   {len(snaps)}')
        cost_file=_WS/'memory'/'cost_log.json'
        if cost_file.exists():
            try:
                data=json.loads(cost_file.read_text())
                print(f'  Cost logs   {len(data)} sessions')
            except: pass
        print()
    elif sub=='clear':
        confirm=input('  Type CLEAR to confirm wiping all memory: ')
        if confirm.strip()=='CLEAR':
            import shutil
            shutil.rmtree(mem_dir,ignore_errors=True)
            mem_dir.mkdir(exist_ok=True)
            print(_g('  Memory cleared.'))
        else:
            print('  Aborted.')
    else:
        print(f'  Unknown sub-command: {sub}')
        print('  Sub-commands: list  stats  clear')

def cmd_setup():
    """Interactive setup wizard — writes / updates config.toml."""
    import getpass
    cfg_path = _WS / 'config.toml'
    print(BANNER)
    print(f'  Essence Setup Wizard\n  {"="*44}')
    print('  Press Enter to accept the default shown in [brackets].\n')

    cfg = _cfg()
    inf = cfg.get('inference', {})
    gov = cfg.get('governance', {})
    ide = cfg.get('identity', {})

    # ── 1. Ollama URL ────────────────────────────────────────────
    cur_ollama = inf.get('ollama_url', 'http://localhost:11434')
    raw = input(f'  Ollama URL [{cur_ollama}]: ').strip()
    ollama_url = raw or cur_ollama
    tags = []
    try:
        import httpx as _hx
        r = _hx.get(f'{ollama_url}/api/tags', timeout=3)
        tags = [m['name'] for m in r.json().get('models', [])]
        print(f'  {_g("✓")} Ollama online — {len(tags)} model(s) available')
    except Exception:
        print(f'  {_r("✗")} Cannot reach Ollama at {ollama_url}')

    # ── 2. Model selection ───────────────────────────────────────
    cur_model = inf.get('model', 'qwen3:4b')
    selected_model = cur_model
    if tags:
        print(f'\n  Available models:')
        shown = tags[:10]
        for i, t in enumerate(shown, 1):
            mark = _c(' ◄ current') if t == cur_model else ''
            print(f'    [{i}] {t}{mark}')
        print(f'    [0] Enter name manually')
        raw = input(f'  Model [{cur_model}]: ').strip()
        if raw:
            try:
                idx = int(raw)
                if idx == 0:
                    selected_model = input('  Model name: ').strip() or cur_model
                elif 1 <= idx <= len(shown):
                    selected_model = shown[idx - 1]
                else:
                    selected_model = raw
            except ValueError:
                selected_model = raw
    else:
        raw = input(f'  Model [{cur_model}]: ').strip()
        selected_model = raw or cur_model

    # ── 3. Performance tier ──────────────────────────────────────
    tier_opts = ['lite', 'standard', 'mem-opt', 'gpu-acc']
    try:
        import psutil as _ps
        ram_gb = _ps.virtual_memory().total / 1024**3
        auto_tier = 'lite' if ram_gb < 8 else 'standard' if ram_gb < 16 else 'mem-opt' if ram_gb < 32 else 'gpu-acc'
        print(f'\n  Detected RAM: {ram_gb:.1f} GB → suggested tier: {auto_tier}')
    except ImportError:
        auto_tier = 'standard'
    cur_tier = inf.get('perf_tier', auto_tier)
    print(f'  Tiers: lite(<8GB)  standard(8-16GB)  mem-opt(16-32GB)  gpu-acc(32GB+)')
    raw = input(f'  Performance tier [{cur_tier}]: ').strip()
    perf_tier = raw if raw in tier_opts else cur_tier

    # ── 4. HuggingFace token ─────────────────────────────────────
    hf_set = bool(os.environ.get('ESSENCE_HF_TOKEN') or os.environ.get('HF_TOKEN'))
    hf_hint = _g('(already set in environment)') if hf_set else _d('(not set)')
    print(f'\n  HuggingFace token {hf_hint}')
    print(f'  Required for gated / private models. Get one at: https://huggingface.co/settings/tokens')
    raw = ''
    try:
        raw = getpass.getpass('  HF token (leave blank to skip): ').strip()
    except Exception:
        pass
    hf_token = raw

    # ── 5. Identity / persona name ───────────────────────────────
    cur_name = ide.get('name', 'Essence')
    raw = input(f'\n  AI persona name [{cur_name}]: ').strip()
    persona_name = raw or cur_name

    # ── 6. Quiet window ──────────────────────────────────────────
    cur_qw_start = gov.get('quiet_window_start', 22)
    cur_qw_end   = gov.get('quiet_window_end', 7)
    print(f'\n  Quiet window — autonomous proactive actions are blocked during these hours.')
    raw = input(f'  Quiet start hour (0-23) [{cur_qw_start}]: ').strip()
    qw_start = int(raw) if raw.isdigit() else cur_qw_start
    raw = input(f'  Quiet end   hour (0-23) [{cur_qw_end}]: ').strip()
    qw_end = int(raw) if raw.isdigit() else cur_qw_end

    # ── Write config.toml ────────────────────────────────────────
    lines = [
        '# Essence configuration — generated by setup wizard',
        f'# Updated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '',
        '[inference]',
        f'ollama_url = "{ollama_url}"',
        f'model      = "{selected_model}"',
        f'perf_tier  = "{perf_tier}"',
        '',
        '[identity]',
        f'name = "{persona_name}"',
        '',
        '[governance]',
        f'quiet_window_start = {qw_start}',
        f'quiet_window_end   = {qw_end}',
        '',
    ]
    if hf_token:
        lines += [
            '[secrets]',
            '# Store HF token here OR set ESSENCE_HF_TOKEN environment variable',
            f'hf_token = "{hf_token}"',
            '',
        ]
    cfg_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\n  {_g("✓")} Config written to {cfg_path}')
    if hf_token:
        print(f'  {_g("✓")} HuggingFace token stored in config.toml')
        print(f'  {_d("  Tip: set ESSENCE_HF_TOKEN in your shell to avoid storing it in the file")}')
    print(f'\n  Setup complete. Run:  python essence.py tui\n')


def cmd_plugin(args):
    """Manage Essence plugins (external SkillAgent files)."""
    plugins_dir = _WS / 'server' / 'skills' / 'plugins'
    plugins_dir.mkdir(parents=True, exist_ok=True)
    sub = args[0] if args else 'list'

    if sub == 'list':
        files = sorted(plugins_dir.glob('*.py'))
        print(f'\n  Essence Plugins ({len(files)})\n  {"="*60}')
        if not files:
            print('  No plugins installed.')
            print('  Install from a URL:  python essence.py plugin install <https://...>')
            print('  Install from file:   python essence.py plugin install path/to/skill.py')
            print()
            return
        print(f'  {"FILE":<36} {"SIZE":>8}  {"MODIFIED"}')
        print('  ' + '-'*58)
        from datetime import datetime as _dt
        for f in files:
            sz  = f.stat().st_size
            mod = _dt.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d')
            print(f'  {f.name:<36} {sz:>8} B  {mod}')
        print(f'\n  {len(files)} plugin(s)  ·  directory: {plugins_dir}')
        print()

    elif sub == 'install':
        source = args[1] if len(args) > 1 else ''
        if not source:
            print(_r('Usage: python essence.py plugin install <https://url> | <local-path>'))
            sys.exit(1)

        src_path = Path(source)
        if src_path.exists() and src_path.suffix == '.py':
            import shutil as _sh
            dest = plugins_dir / src_path.name
            _sh.copy2(src_path, dest)
            print(f'  {_g("Installed:")} {dest.name}')
            return

        if source.startswith(('http://', 'https://')):
            url = source
            if 'github.com' in url and '/blob/' in url:
                url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            try:
                import httpx as _hx
                r = _hx.get(url, timeout=30, follow_redirects=True)
                r.raise_for_status()
                filename = url.rstrip('/').split('/')[-1]
                if not filename.endswith('.py'):
                    filename += '.py'
                dest = plugins_dir / filename
                dest.write_bytes(r.content)
                print(f'  {_g("Installed:")} {dest.name}  ({len(r.content):,} bytes)')
            except Exception as e:
                print(_r(f'  Install failed: {e}'))
                sys.exit(1)
        else:
            print(_r('  Source must be a .py file path or an https:// URL.'))
            sys.exit(1)

    elif sub == 'uninstall':
        name = args[1] if len(args) > 1 else ''
        if not name:
            print(_r('Usage: python essence.py plugin uninstall <filename>'))
            sys.exit(1)
        if not name.endswith('.py'):
            name += '.py'
        p = plugins_dir / name
        if p.exists():
            p.unlink()
            print(f'  {_g("Uninstalled:")} {name}')
        else:
            print(_r(f'  Not found: {name}'))
            sys.exit(1)

    elif sub == 'update':
        name = args[1] if len(args) > 1 else ''
        if not name:
            print(_r('Usage: python essence.py plugin update <name>'))
            sys.exit(1)
        print(f'  To update a plugin, re-install it from the original source:')
        print(f'  python essence.py plugin install <url-or-path>')

    else:
        print(f'  Unknown sub-command: {sub}')
        print('  Sub-commands: list  install <url|path>  uninstall <name>  update <name>')


def cmd_automate(args):
    """Manage automation patterns and proactive triggers."""
    sys.path.insert(0, str(_WS))
    sub = args[0] if args else 'list'

    try:
        from server.action_engine import PatternRegistry, Pattern, PatternState, TriggerType
        from server.action_engine import _DEFAULT_DB as _pat_db
        reg = PatternRegistry(_pat_db)
    except ImportError as e:
        print(_r(f'  action_engine not available: {e}'))
        sys.exit(1)

    if sub == 'list':
        patterns = reg.all_patterns()
        print(f'\n  Automation Patterns ({len(patterns)})\n  {"="*66}')
        if not patterns:
            print('  No patterns defined.')
            print('  Add one:  python essence.py automate add --name "Daily brief" \\')
            print('              --trigger cron --cron "0 9 * * *" --action "give me my daily summary"')
            print()
            return
        print(f'  {"ID":<10} {"NAME":<24} {"TRIGGER":<10} {"STATE":<12} {"FIRES":>6}')
        print('  ' + '-'*66)
        for p in patterns:
            trig  = p.trigger_type
            fires = str(p.fire_count)
            state_col = _g(p.state) if p.state == PatternState.ACTIVE else \
                        _r(p.state) if p.state == PatternState.ARCHIVED else \
                        _d(p.state)
            print(f'  {p.id:<10} {p.name:<24} {trig:<10} {state_col:<20} {fires:>6}')
        print()

    elif sub == 'add':
        def _arg(flag):
            try: idx = args.index(flag); return args[idx + 1]
            except (ValueError, IndexError): return ''
        name        = _arg('--name')
        trigger     = _arg('--trigger') or TriggerType.TIME
        cron_expr   = _arg('--cron') or '0 9 * * *'
        action      = _arg('--action')
        autonomy    = float(_arg('--autonomy') or '0.6')
        event_topic = _arg('--event')
        condition   = _arg('--condition')
        desc        = _arg('--desc') or ''

        if not name or not action:
            print(_r('Usage: python essence.py automate add --name <name> --action <intent> [options]'))
            print('  Options:')
            print('    --trigger  time|event|condition     (default: time)')
            print('    --cron     "0 9 * * *"              (cron expression for time trigger)')
            print('    --event    <bus-topic>              (for event trigger)')
            print('    --condition <name>                  (for condition trigger)')
            print('    --autonomy 0.0-1.0                  (default: 0.6 → suggest)')
            print('    --desc     "description"')
            sys.exit(1)

        import uuid as _uuid
        trigger_spec = {}
        if trigger == TriggerType.TIME:
            trigger_spec = {"cron": cron_expr}
        elif trigger == TriggerType.EVENT:
            trigger_spec = {"topic": event_topic}
        elif trigger == TriggerType.CONDITION:
            trigger_spec = {"condition": condition}

        p = Pattern(
            id=_uuid.uuid4().hex[:8],
            name=name,
            description=desc,
            trigger_type=trigger,
            trigger_spec=trigger_spec,
            action_spec={"intent": action, "autonomy_level": autonomy},
            state=PatternState.ACTIVE,
            gravity=autonomy,
        )
        reg.upsert(p)
        print(f'  {_g("Added:")} {p.id} — {name}  [{trigger}]')

    elif sub == 'rm':
        pid = args[1] if len(args) > 1 else ''
        if not pid:
            print(_r('Usage: python essence.py automate rm <id>'))
            sys.exit(1)
        p = reg.get(pid)
        if not p:
            print(_r(f'  Pattern not found: {pid}'))
            sys.exit(1)
        reg.transition(pid, PatternState.ARCHIVED)
        print(f'  {_g("Archived:")} {pid} ({p.name})')

    elif sub == 'activate':
        pid = args[1] if len(args) > 1 else ''
        if not pid:
            print(_r('Usage: python essence.py automate activate <id>'))
            sys.exit(1)
        reg.transition(pid, PatternState.ACTIVE)
        print(f'  {_g("Activated:")} {pid}')

    elif sub == 'run':
        pid = args[1] if len(args) > 1 else ''
        if not pid:
            print(_r('Usage: python essence.py automate run <id>'))
            sys.exit(1)
        p = reg.get(pid)
        if not p:
            print(_r(f'  Pattern not found: {pid}'))
            sys.exit(1)
        intent = p.action_spec.get('intent', '')
        print(f'\n  Pattern : {p.name}')
        print(f'  Intent  : {intent}')
        print(f'\n  Firing via chat:')
        # Run intent through simple chat
        import httpx as _hx
        model, base = _model(), _get_ollama()
        soul = _WS / 'SOUL.md'
        msgs = []
        if soul.exists():
            msgs.append({'role': 'system', 'content': soul.read_text(encoding='utf-8')})
        msgs.append({'role': 'user', 'content': intent})
        try:
            with _hx.Client(timeout=60) as c:
                with c.stream('POST', f'{base}/api/chat',
                              json={'model': model, 'messages': msgs, 'stream': True}) as rr:
                    rr.raise_for_status()
                    for line in rr.iter_lines():
                        if not line: continue
                        try:
                            d = json.loads(line)
                            t = d.get('message', {}).get('content', '')
                            print(t, end='', flush=True)
                            if d.get('done'): break
                        except Exception: continue
            print()
            reg.record_fire(pid)
        except Exception as e:
            print(_r(f'\n  Error: {e}'))

    else:
        print(f'  Unknown sub-command: {sub}')
        print('  Sub-commands: list  add  rm <id>  activate <id>  run <id>')


def cmd_evolve(args):
    """Self-evolve a skill using LLM-as-judge multi-dimensional fitness scoring.

    Usage:
      python essence.py evolve list              -- list evolvable skills
      python essence.py evolve <skill_id>        -- run evolution (3 iterations)
      python essence.py evolve <skill_id> --iter N     -- custom iteration count
      python essence.py evolve <skill_id> --dry-run    -- validate setup only
      python essence.py evolve <skill_id> --model qwen3:8b  -- override model

    Evolved output is NEVER auto-deployed. It is saved to:
      memory/skills/evolved/<skill_id>_v<N>.md  + _metrics.json

    Review the metrics and promote manually if the evolved version is better.
    """
    import asyncio as _aio
    sys.path.insert(0, str(_WS))
    skills_dir = _WS / 'memory' / 'skills'
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Parse args  (skip flag args when picking the sub-command/skill-id)
    positional = [a for a in args if not a.startswith('--')]
    sub        = positional[0] if positional else 'list'
    dry_run    = '--dry-run' in args
    model      = _get_model()
    iterations = 3

    for i, a in enumerate(args):
        if a == '--iter' and i + 1 < len(args):
            try: iterations = int(args[i+1])
            except ValueError: pass
        if a == '--model' and i + 1 < len(args):
            model = args[i + 1]

    if sub == 'list':
        from server.skill_evolution import list_evolvable_skills
        skills = list_evolvable_skills(skills_dir)
        if not skills:
            print(f'  No skills found in {skills_dir}')
            print(f'  Add skills with: python essence.py skills add <id>')
            return
        print(f'\n  {_b("Evolvable skills")}  ({len(skills)} total)\n')
        print(f'  {"ID":<30}  {"Chars":>7}')
        print(f'  {"─"*30}  {"─"*7}')
        for s in skills:
            print(f'  {s["id"]:<30}  {s["chars"]:>7,}')
        # Also list evolved versions
        evolved_dir = skills_dir / 'evolved'
        if evolved_dir.exists():
            evolved = list(evolved_dir.glob('*_metrics.json'))
            if evolved:
                print(f'\n  {_d("Evolved versions:")} {len(evolved)}')
                for m in evolved[:5]:
                    try:
                        data = json.loads(m.read_text())
                        delta = data.get('evolved', {}).get('delta', 0)
                        improved = data.get('improved', False)
                        icon = _g('↑') if improved else _d('↔')
                        print(f'    {icon} {m.stem.replace("_metrics",""):<35} {_g("+"+str(round(delta,3))) if improved else _d(str(round(delta,3)))}')
                    except Exception:
                        pass
        print()
        return

    # Treat sub as skill_id
    skill_id = sub
    print(f'\n  {_b("Essence Skill Evolution")} — skill: {_c(skill_id)}\n')
    print(f'  Model:       {model}')
    print(f'  Iterations:  {iterations}')
    print(f'  Dry run:     {dry_run}')
    print()

    async def _run():
        from server.skill_evolution import evolve_skill
        result = await evolve_skill(
            skill_id=skill_id,
            skills_dir=skills_dir,
            ollama_url=_get_ollama(),
            model=model,
            iterations=iterations,
            dry_run=dry_run,
        )
        return result

    try:
        result = _aio.run(_run())
    except FileNotFoundError as exc:
        print(f'  {_r("Error:")} {exc}')
        print(f'  Available skills: python essence.py evolve list')
        return
    except Exception as exc:
        print(f'  {_r("Evolution failed:")} {exc}')
        import traceback; traceback.print_exc()
        return

    if result.get('dry_run'):
        print(f'  {_g("Dry run passed.")}')
        print(f'  Skill: {result["skill_id"]} ({result["skill_chars"]:,} chars)')
        print(f'  Would generate {result["would_generate"]} eval examples, run {result["would_iterate"]} iterations')
        return

    improved = result.get('improved', False)
    baseline = result.get('baseline_score', 0)
    evolved  = result.get('evolved_score', 0)
    delta    = result.get('delta', 0)

    print(f'  Baseline score:  {baseline:.3f}')
    print(f'  Evolved score:   {evolved:.3f}  ({_g("+"+str(round(delta,3))) if delta > 0 else _r(str(round(delta,3)))})')
    print()
    if improved:
        print(f'  {_g("✓ Evolution succeeded!")}  Composite score improved by {delta:.3f}')
    else:
        print(f'  {_d("No significant improvement")} (Δ{delta:.3f} < threshold)')
    print()
    print(f'  Output:   {result.get("output_path","?")}')
    print(f'  Metrics:  {result.get("metrics_path","?")}')
    print()
    if improved:
        print(f'  {_d("Review the evolved skill and promote manually:")}')
        print(f'  cp {result.get("output_path","?")} memory/skills/{skill_id}.md')
    print()


def cmd_insights(args):
    """Print session analytics for the past N days.

    Usage:
      python essence.py insights         -- last 30 days (default)
      python essence.py insights 7       -- last 7 days
      python essence.py insights 90      -- last 90 days
    """
    sys.path.insert(0, str(_WS))
    days = 30
    if args:
        try:
            days = int(args[0])
        except ValueError:
            print(_r(f'  Invalid days value: {args[0]}'))
            return

    from server.session_insights import get_insights_engine
    engine = get_insights_engine(_WS)
    print(f'  Generating insights for last {days} days …')
    report = engine.generate(days=days)
    print(engine.format_terminal(report))


def cmd_pool(args):
    """Manage the multi-credential pool for provider key rotation.

    Usage:
      python essence.py pool list [provider]        -- show pool state
      python essence.py pool add  <provider> <key> [label]  -- add a key
      python essence.py pool reset [provider]       -- clear exhaustion
      python essence.py pool rotate [provider]      -- force rotate to next key
    """
    sys.path.insert(0, str(_WS))
    from server.credential_pool import get_pool

    sub      = args[0] if args else 'list'
    provider = (args[1] if len(args) > 1 else None) or 'openai'

    if sub == 'list':
        pool = get_pool(provider)
        print(f'\n  Credential Pool — provider: {provider}')
        for line in pool.summary_lines():
            print(line)
        print(f'\n  Add a key:    python essence.py pool add {provider} <key> [label]')
        print(f'  Reset all:    python essence.py pool reset {provider}')
        print(f'  Rotate:       python essence.py pool rotate {provider}')
        print()

    elif sub == 'add':
        # essence.py pool add <provider> [label]
        # Key is read via getpass — never passed as a CLI argument (avoids shell history)
        provider = args[1] if len(args) > 1 else 'openai'
        label    = args[2] if len(args) > 2 else ''
        import getpass as _gp
        api_key = _gp.getpass(f'  API key for {provider} (hidden): ').strip()
        if not api_key:
            print(_r('  No key entered — aborted.'))
            return
        pool  = get_pool(provider)
        entry = pool.add(api_key, label=label or f'key-{len(pool.entries())+1}')
        print(_g(f'  Added credential {entry.id[:6]} ({entry.label}) to pool[{provider}]'))

    elif sub == 'reset':
        pool  = get_pool(provider)
        count = pool.reset_all()
        print(_g(f'  Reset {count} exhausted credential(s) in pool[{provider}]'))

    elif sub == 'rotate':
        pool = get_pool(provider)
        if len(pool.entries()) < 2:
            print(_d('  Pool has fewer than 2 credentials — nothing to rotate to.'))
            return
        nxt = pool.rotate()
        if nxt:
            print(_g(f'  Rotated to: {nxt.short_label()} [{nxt.id[:6]}]'))
        else:
            print(_r('  All credentials exhausted — no rotation available.'))

    else:
        print(_r(f'  Unknown subcommand: {sub}'))
        print('  Usage: python essence.py pool list|add|reset|rotate')


def cmd_services(args):
    """Manage learned external services / APIs."""
    import asyncio as _aio
    sys.path.insert(0, str(_WS))
    data_dir = _WS / 'data'

    try:
        from server.service_registry import init_service_registry
        from server.service_ingestor import init_ingestor
        reg = init_service_registry(data_dir / 'service_registry.db')
        init_ingestor(ollama_url=_get_ollama(), model=_get_model())
    except ImportError as e:
        print(_r(f'  Service registry unavailable: {e}')); sys.exit(1)

    sub = args[0] if args else 'list'

    if sub == 'list':
        svcs = reg.list_all()
        print(f'\n  Learned Services ({len(svcs)})\n  {"="*70}')
        if not svcs:
            print('  No services ingested yet.')
            print('  Ingest one:  python essence.py services ingest <url>')
            print()
            return
        print(f'  {"ID":<28} {"NAME":<28} {"ENDPOINTS":>9}  {"BASE URL"}')
        print('  ' + '-'*80)
        from datetime import datetime as _dt
        for s in svcs:
            ts = _dt.fromtimestamp(s['updated_at']).strftime('%Y-%m-%d')
            print(f'  {s["id"]:<28} {s["name"]:<28} {s["endpoints"]:>9}  {s["base_url"][:40]}')
        print()

    elif sub == 'ingest':
        url = args[1] if len(args) > 1 else ''
        if not url:
            print(_r('Usage: python essence.py services ingest <url>')); sys.exit(1)
        print(f'  Ingesting {url} …')
        try:
            from server.service_ingestor import get_ingestor
            profile = _aio.run(get_ingestor().ingest(url))
            print(f'  {_g("Done.")} {profile.name}  ({len(profile.endpoints)} endpoints)')
            print(f'  Service id: {_c(profile.id)}')
            print(f'  Base URL:   {profile.base_url}')
        except Exception as e:
            print(_r(f'  Ingestion failed: {e}')); sys.exit(1)

    elif sub == 'show':
        sid = args[1] if len(args) > 1 else ''
        if not sid:
            print(_r('Usage: python essence.py services show <id>')); sys.exit(1)
        profile = reg.get(sid)
        if not profile:
            print(_r(f'  Service not found: {sid}')); sys.exit(1)
        print(f'\n  {profile.name}  ({profile.id})')
        print(f'  Base URL:    {profile.base_url}')
        print(f'  Description: {profile.description}')
        print(f'  Auth:        {profile.auth.type}')
        print(f'  Source:      {profile.source_url}')
        print(f'\n  Endpoints ({len(profile.endpoints)}):')
        print(f'  {"METHOD":<8} {"PATH":<40} {"DESCRIPTION"}')
        print('  ' + '-'*80)
        for ep in profile.endpoints:
            p_list = ', '.join(f'{p.name}{"*" if p.required else ""}' for p in ep.params[:4])
            print(f'  {ep.method:<8} {ep.path:<40} {ep.description[:36]}')
            if p_list:
                print(f'           params: {p_list}')
        print()

    elif sub == 'forget':
        sid = args[1] if len(args) > 1 else ''
        if not sid:
            print(_r('Usage: python essence.py services forget <id>')); sys.exit(1)
        ok = reg.delete(sid)
        if ok:
            print(f'  {_g("Removed:")} {sid}')
        else:
            print(_r(f'  Service not found: {sid}')); sys.exit(1)

    elif sub == 'auth':
        # python essence.py services auth <id> <key> <value>
        if len(args) < 4:
            print(_r('Usage: python essence.py services auth <id> <key> <value>'))
            print('  e.g.  python essence.py services auth my-api api_key sk-...')
            sys.exit(1)
        sid, key, value = args[1], args[2], args[3]
        ok = reg.set_cred(sid, key, value)
        if ok:
            masked = value[:4] + '...' + value[-4:] if len(value) > 12 else '***'
            print(f'  {_g("Set")} {sid}.{key} = {masked}')
        else:
            print(_r(f'  Service not found: {sid}')); sys.exit(1)

    elif sub == 'search':
        query = ' '.join(args[1:]) if len(args) > 1 else ''
        if not query:
            print(_r('Usage: python essence.py services search <query>')); sys.exit(1)
        results = reg.search(query)
        print(f'\n  Services matching "{query}"\n  {"="*44}')
        if not results:
            print('  No matches.')
        for r in results:
            print(f'  {_c(r["id"])}  {r["name"]}  ({r["base_url"]})')
        print()

    else:
        print(f'  Unknown sub-command: {sub}')
        print('  Sub-commands: list  ingest <url>  show <id>  forget <id>  auth <id> <key> <val>  search <q>')


def cmd_vault(args):
    """Manage the Essence credential vault.

    Usage:
      python essence.py vault list                           -- list providers
      python essence.py vault store <provider> <env_var>    -- store API key (prompted)
      python essence.py vault delete <provider>             -- remove a key
    """
    sys.path.insert(0, str(_WS))
    try:
        from server.credential_vault import get_vault
    except ImportError as e:
        print(_r(f'  credential_vault not available: {e}')); sys.exit(1)

    vault = get_vault()
    sub   = args[0] if args else 'list'

    if sub == 'list':
        providers = vault.list_providers()
        print(f'\n  Credential Vault ({len(providers)} provider(s))\n  {"="*44}')
        if not providers:
            print('  No keys stored. Use: python essence.py vault store <provider> <env_var>')
        for p in providers:
            print(f'  {_g("✓")} {p}')
        print()

    elif sub == 'store':
        provider = args[1] if len(args) > 1 else ''
        env_var  = args[2] if len(args) > 2 else ''
        if not provider or not env_var:
            print(_r('Usage: python essence.py vault store <provider> <env_var>')); sys.exit(1)
        import getpass as _gp
        api_key = _gp.getpass(f'  API key for {provider} ({env_var}) [hidden]: ').strip()
        if not api_key:
            print(_r('  No key entered — aborted.')); return
        vault.store(provider, env_var, api_key)
        print(f'  {_g("Stored:")} {provider} ({env_var})')

    elif sub == 'delete':
        provider = args[1] if len(args) > 1 else ''
        if not provider:
            print(_r('Usage: python essence.py vault delete <provider>')); sys.exit(1)
        ok = vault.delete(provider)
        print(f'  {_g("Deleted:")} {provider}' if ok else _r(f'  Not found: {provider}'))

    else:
        print(f'  Unknown sub-command: {sub}')
        print('  Sub-commands: list  store <provider> <env_var>  delete <provider>')


def cmd_clean(args):
    """Housekeeping: vacuum SQLite DBs, prune episodic turns, archive old logs."""
    import shutil as _sh
    from datetime import datetime as _dt, timedelta as _td
    import sqlite3 as _sq

    data_dir = _WS / 'data'
    sub = args[0] if args else 'all'

    if sub == '--help':
        print('  Sub-commands: all  vacuum  episodic  logs  cache')
        print('  Example:      python essence.py clean all')
        return

    print(f'\n  Essence Housekeeping\n  {"="*44}')

    def _vacuum(path: Path, label: str):
        if not path.exists():
            print(f'  {label}: not found, skipping')
            return
        before = path.stat().st_size
        try:
            conn = _sq.connect(str(path))
            conn.execute('VACUUM')
            conn.close()
            after = path.stat().st_size
            saved = max(0, before - after)
            if saved > 0:
                print(f'  {_g("Vacuumed")} {label}: freed {saved // 1024} KB')
            else:
                print(f'  {label}: already compact')
        except Exception as e:
            print(_r(f'  Vacuum {label} failed: {e}'))

    if sub in ('all', 'vacuum'):
        print('\n  SQLite vacuum:')
        for dbname in ('episodic.db', 'gravity_memory.db', 'event_log.db', 'patterns.db'):
            _vacuum(data_dir / dbname, dbname)

    if sub in ('all', 'episodic'):
        print('\n  Episodic memory prune:')
        ep_db = data_dir / 'episodic.db'
        if ep_db.exists():
            try:
                conn = _sq.connect(str(ep_db))
                total = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE archived = 0"
                ).fetchone()[0]
                keep  = 500
                if total > keep:
                    conn.execute("""
                        UPDATE episodes SET archived = 1
                        WHERE id IN (
                            SELECT id FROM episodes
                            WHERE archived = 0
                            ORDER BY ts ASC
                            LIMIT ?
                        )
                    """, (total - keep,))
                    conn.commit()
                    print(f'  Archived {total - keep} old turns (kept {keep} most-recent)')
                else:
                    print(f'  {total} active episodic turns — no pruning needed')
                conn.close()
            except Exception as e:
                print(_r(f'  Episodic prune failed: {e}'))
        else:
            print('  episodic.db not found, skipping')

    if sub in ('all', 'logs'):
        print('\n  Log archive:')
        cutoff      = _dt.now() - _td(days=7)
        archive_dir = _WS / 'logs' / 'archive'
        archived    = 0
        for lf in _WS.glob('*.log'):
            if _dt.fromtimestamp(lf.stat().st_mtime) < cutoff:
                archive_dir.mkdir(parents=True, exist_ok=True)
                stamp = _dt.fromtimestamp(lf.stat().st_mtime).strftime('%Y%m%d')
                dest  = archive_dir / f'{lf.stem}_{stamp}.log'
                _sh.move(str(lf), str(dest))
                archived += 1
                print(f'  Archived {lf.name} → logs/archive/')
        if not archived:
            print('  No log files older than 7 days')

    if sub in ('all', 'cache'):
        print('\n  Python cache cleanup:')
        cleared = 0
        for cache_dir in [_WS / '__pycache__', _WS / 'server' / '__pycache__',
                          _WS / 'server' / 'skills' / '__pycache__']:
            if cache_dir.exists():
                _sh.rmtree(cache_dir, ignore_errors=True)
                cleared += 1
                print(f'  Cleared {cache_dir.relative_to(_WS)}')
        if not cleared:
            print('  No __pycache__ directories found')

    print(f'\n  {_g("Housekeeping complete.")}\n')


def main():
    av=sys.argv[1:]; cmd=av[0] if av else ''
    if not cmd or cmd=='probe': cmd_probe()
    elif cmd=='install': cmd_install()
    elif cmd=='version': cmd_version()
    elif cmd=='pull':
        if len(av)<2: print(_r('Usage: python essence.py pull <model>')); sys.exit(1)
        cmd_pull(av[1])
    elif cmd=='chat': cmd_chat()
    elif cmd in ('tui', 'up'): cmd_tui()  # 'up' is the documented quickstart alias
    elif cmd=='skills': cmd_skills(av[1:])
    elif cmd=='mcp': cmd_mcp(av[1:])
    elif cmd=='models': cmd_models(av[1:])
    elif cmd=='logs': cmd_logs(av[1:])
    elif cmd=='cost': cmd_cost(av[1:])
    elif cmd=='snapshot': cmd_snapshot(av[1:])
    elif cmd=='cron': cmd_cron(av[1:])
    elif cmd=='health': cmd_health()
    elif cmd=='config': cmd_config(av[1:])
    elif cmd=='memory': cmd_memory(av[1:])
    elif cmd=='setup': cmd_setup()
    elif cmd=='plugin': cmd_plugin(av[1:])
    elif cmd=='automate': cmd_automate(av[1:])
    elif cmd=='services': cmd_services(av[1:])
    elif cmd=='evolve': cmd_evolve(av[1:])
    elif cmd=='insights': cmd_insights(av[1:])
    elif cmd=='pool': cmd_pool(av[1:])
    elif cmd in ('clean','housekeeping'): cmd_clean(av[1:])
    elif cmd=='vault': cmd_vault(av[1:])
    else:
        print(f'Unknown command: {cmd}')
        print('Commands:')
        print()
        print('  Core')
        print('  ────────────────────────────────────────────────')
        print('  probe          System probe & Ollama status')
        print('  install        Install Python dependencies')
        print('  setup          Interactive setup wizard')
        print('  tui            Full-featured TUI (recommended)')
        print('  chat           Simple streaming chat REPL')
        print()
        print('  Models')
        print('  ────────────────────────────────────────────────')
        print('  pull <model>   Pull an Ollama model')
        print('  models         Ollama: list / pull / rm / select')
        print('                 HuggingFace: hf-list / hf-pull / hf-run')
        print()
        print('  Automation & Skills')
        print('  ────────────────────────────────────────────────')
        print('  skills         Manage skill agents (list add enable disable remove)')
        print('  evolve         Self-evolve a skill via LLM-as-judge (evolve list | evolve <id>)')
        print('  plugin         Manage external plugins (list install uninstall)')
        print('  automate       Proactive patterns (list add rm activate run)')
        print('  cron           Cron jobs (list add enable disable rm)')
        print()
        print('  Memory & Data')
        print('  ────────────────────────────────────────────────')
        print('  memory         Memory browser (list stats clear)')
        print('  snapshot       Session snapshots (list save rm)')
        print('  cost           Token & cost summary')
        print('  insights [N]   Session analytics: tokens, cost, tools, activity (last N days)')
        print('  pool           Multi-key credential pool (list add reset rotate)')
        print('  clean          Housekeeping: vacuum DBs, prune episodic, archive logs')
        print()
        print('  Services & Integrations')
        print('  ────────────────────────────────────────────────')
        print('  services       Learned APIs (list ingest show forget auth search)')
        print('  mcp            MCP servers (list add remove enable disable)')
        print('  vault          Credential vault (list store delete)')
        print()
        print('  System')
        print('  ────────────────────────────────────────────────')
        print('  logs           Show recent log lines [--tail N]')
        print('  health         Quick system health check')
        print('  config         Show or query config (show get <key>)')
        print('  version        Show version info')
        sys.exit(1)
if __name__=='__main__': main()
