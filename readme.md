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
python run_agent.py
```

### Analyze failures
```bash
python run_judge.py --all-traces --definitions MAST/taxonomy_definitions_examples/definitions.txt --outdir mast_agent/saved_results
```

## Architecture

- **Search Node**: TavilySearch for information retrieval
- **Reason Node**: Claude-4-Sonnet for reasoning and query reformulation
- **Memory**: LangGraph MemorySaver for persistent context
- **Loop**: Conditional edges based on follow-up queries

## Files

- `agent.py` - LangGraph multi-agent system
- `sample_dataset.py` - Sample random HOTPOT-QA questions
- `run_agent.py` - Run the agent on random HOTPOT-QA questions
- `run_judge.py` - MAST failure mode analysis
- `deliverable.json` - Final analysis results
- `output/` - Agent execution traces
- `mast_agent/saved_results/` - Judge evaluation results
