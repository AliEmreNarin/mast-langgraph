# MAST Agent Analysis

Multi-agent system failure analysis using LangGraph and HOTPOT-QA dataset.

## Setup
```bash
pip install -r requirements.txt
```

Set environment variables:
- `ANTHROPIC_API_KEY`
- `TAVILY_API_KEY` 
- `OPENAI_API_KEY`

## Usage

### Generate traces
```bash
python main.py
```

### Analyze failures
```bash
python run_judge.py --all-traces --definitions MAST/taxonomy_definitions_examples/definitions.txt --outdir mast_agent/saved_results
```

## Architecture

- **Search Node**: TavilySearch for information retrieval
- **Reason Node**: Claude-3-Haiku for reasoning and query reformulation
- **Memory**: LangGraph MemorySaver for persistent context
- **Loop**: Conditional edges based on follow-up queries

## Files

- `agent.py` - LangGraph multi-agent system
- `main.py` - Run 5 HOTPOT-QA questions
- `run_judge.py` - MAST failure mode analysis
- `deliverable.json` - Final analysis results
- `output/` - Agent execution traces
- `mast_agent/saved_results/` - Judge evaluation results

## Results

Analyzed 50 traces, identified common failure modes:
- Task specification violations
- Weak verification
- Ambiguity handling failures
- Format compliance issues