# Refactoring Decision & Planning Agent (RDP Agent)

A Python-based agent that analyses code quality reports and produces structured refactoring plans. Part of a multi-agent refactoring system.

## Architecture

```
Code Understanding Agent ──► RDP Agent ──► Safe Transformation Agent
     (quality_report.json)   (this repo)    (refactoring_plan.json)
```

The RDP Agent:
1. **Parses** a quality report containing detected code smells and metrics
2. **Selects** the best refactoring for each smell using a rule-based scoring engine
3. **Sequences** the refactorings respecting dependency constraints
4. **Generates** a structured JSON plan with human-readable explanations

## Quick Start

### Prerequisites

- Python 3.10+
- Flask (for the Web UI): `pip install flask`
- (Optional) PyYAML for YAML config support: `pip install pyyaml`
- (Optional) pytest for running tests: `pip install pytest`

Install all dependencies at once:

```bash
pip install flask pyyaml pytest
```

### Option 1 — Web UI (Recommended)

Start the Flask development server:

```bash
python app.py
```

Then open your browser and navigate to:

```
http://localhost:5000
```

From the web interface you can:

1. **Drag & drop** (or browse) a `.json` quality report file
2. Click **Generate Refactoring Plan**
3. View the plan summary and individual refactoring steps
4. **Copy** the JSON to clipboard or **Download** it as a file

### Option 2 — Command Line

#### Run with Sample Data

```bash
python rdp_agent.py --input quality_report.json --output refactoring_plan.json
```

#### Run with Custom Config

```bash
python rdp_agent.py --input quality_report.json --output plan.json --config config.yaml
```

### Run Tests

```bash
pytest test_rdp_agent.py -v
```

## Project Structure

```
├── app.py                 # Flask web server (Web UI entry point)
├── rdp_agent.py           # Main agent (all core logic)
├── quality_report.json    # Sample input from Code Understanding Agent
├── config.yaml            # Configurable weights, thresholds, log level
├── test_rdp_agent.py      # pytest test suite
├── templates/
│   └── index.html         # Web UI template (upload form & results)
├── static/
│   └── style.css          # Web UI styles (dark theme)
└── README.md              # This file
```

## Input Format

The agent expects a JSON file with the following structure:

```json
{
  "target": "OrderProcessor.java",
  "smells": [
    {
      "id": "smell_001",
      "type": "Long Method",
      "location": { "class": "OrderProcessor", "method": "calculateTotal", "lines": [10, 160] },
      "metrics": { "lines_of_code": 150, "cyclomatic_complexity": 30 },
      "severity": "high"
    }
  ],
  "metrics_summary": { "total_lines": 850 }
}
```

## Output Format

The generated plan follows this structure:

```json
{
  "plan_id": "plan_20250321_135600",
  "target": "OrderProcessor.java",
  "steps": [
    {
      "step_id": 1,
      "smell_id": "smell_001",
      "refactoring": "Extract Method",
      "target": { "class": "OrderProcessor", "method": "calculateTotal" },
      "parameters": { "source_lines": [10, 160], "new_method_name": "extracted_calculateTotal" },
      "explanation": "Extract Method on OrderProcessor.calculateTotal to address Long Method smell. ..."
    }
  ],
  "summary": "3-step plan addressing 3 of 7 detected smells..."
}
```

## Supported Smell Types

| Smell Type | Candidate Refactorings |
|---|---|
| Long Method | Extract Method, Replace Temp with Query, Introduce Parameter Object |
| God Class | Extract Class, Extract Subclass |
| Feature Envy | Move Method |
| Duplicate Code | Extract Method, Pull Up Method |
| Data Clumps | Introduce Parameter Object, Extract Class |
| Shotgun Surgery | Move Method, Inline Class |
| Switch Statements | Replace Conditional with Polymorphism |
| Lazy Class | Inline Class, Collapse Hierarchy |
| Speculative Generality | Collapse Hierarchy, Remove Dead Code |
| Primitive Obsession | Replace Data Value with Object, Introduce Parameter Object |
| Long Parameter List | Introduce Parameter Object, Replace Parameter with Method Call |
| Message Chains | Hide Delegate |
| Comments | Extract Method, Rename Method |

## Configuration

Edit `config.yaml` to adjust behaviour without changing code:

```yaml
weights:
  complexity_weight: 0.2  # Lower complexity is better
  risk_weight: 0.4        # Lower risk is better
  impact_weight: 0.4      # Higher impact is better

severity_order:
  critical: 4
  high: 3
  medium: 2
  low: 1

log_level: INFO
```

## Scoring Formula

```
score = complexity_weight × (4 - complexity) + risk_weight × (4 - risk) + impact_weight × impact
```

Where `low=1, medium=2, high=3`. Higher score = better candidate.

## License

MIT