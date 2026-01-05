# Claude Code Subagents for NBA Value Betting Platform

This folder contains specialized subagents for building the NBA Value Betting Platform. Each subagent has domain expertise and specific tool permissions optimized for their role.

## Quick Setup

1. **Copy to your project:**
   ```bash
   # From your project root
   mkdir -p .claude/agents
   cp path/to/claude-agents/*.md .claude/agents/
   ```

2. **Verify agents are loaded:**
   - Start Claude Code in your project directory
   - Run `/agents` to see all available subagents

3. **Use agents:**
   - Claude will automatically delegate to appropriate agents
   - Or explicitly request: "Have the ml-engineer implement the calibration layer"

## Subagents Included

| Agent | Purpose | Tools |
|-------|---------|-------|
| `data-engineer` | Data pipelines, database schemas, ETL | Read, Write, Edit, Bash, Glob, Grep |
| `ml-engineer` | Model training, calibration, scoring algorithms | Read, Write, Edit, Bash, Glob, Grep |
| `backend-engineer` | FastAPI services, API endpoints, auth | Read, Write, Edit, Bash, Glob, Grep |
| `frontend-engineer` | React components, UI/UX, visualization | Read, Write, Edit, Bash, Glob, Grep |
| `test-writer` | Unit tests, integration tests, coverage | Read, Write, Edit, Bash, Glob, Grep |
| `code-reviewer` | Quality review, security audit | Read, Grep, Glob (read-only) |
| `devops-engineer` | Docker, CI/CD, deployment, monitoring | Read, Write, Edit, Bash, Glob, Grep |
| `research-analyst` | API research, technology evaluation | Read, Grep, Glob, WebFetch, WebSearch |

## Recommended Usage by Phase

### Phase 1: Data Infrastructure
```
> Have the data-engineer design the database schema
> Have the data-engineer build the odds ingestion pipeline
> Use the research-analyst to evaluate odds API options
```

### Phase 2: Model Layer
```
> Have the ml-engineer implement the MOV model training pipeline
> Have the ml-engineer build the calibration layer
> Have the test-writer write tests for the scoring algorithms
```

### Phase 3: Scoring Engines
```
> Have the ml-engineer implement Algorithm A (Idea 1 style)
> Have the ml-engineer implement Algorithm B (Idea 2 style)
> Have the code-reviewer verify the probability calculations
```

### Phase 4: API & Backend
```
> Have the backend-engineer create the FastAPI endpoints
> Have the backend-engineer implement authentication
> Have the test-writer write API integration tests
```

### Phase 5: Frontend
```
> Have the frontend-engineer build the Market Board component
> Have the frontend-engineer implement the Bet Detail view
> Have the frontend-engineer add the Value Score visualization
```

### Phase 6: Deployment
```
> Have the devops-engineer create the Docker configuration
> Have the devops-engineer set up the CI/CD pipeline
> Have the devops-engineer configure monitoring
```

## Customizing Agents

Each agent file is a Markdown file with YAML frontmatter:

```markdown
---
name: agent-name
description: When this agent should be used
tools: Read, Write, Edit, Bash, Glob, Grep
---

System prompt content here...
```

### Adding MCP Tools

If you have MCP servers configured, add their tools:

```markdown
---
name: data-engineer
description: Data pipeline tasks
tools: Read, Write, Edit, Bash, Glob, Grep, mcp__postgres__query
---
```

### Adjusting Tool Permissions

- **Read-only agents** (reviewers): `Read, Grep, Glob`
- **Research agents**: Add `WebFetch, WebSearch`
- **Full access**: `Read, Write, Edit, Bash, Glob, Grep`

## Tips for Effective Use

1. **Be specific when delegating:**
   - ✅ "Have the ml-engineer implement isotonic calibration for spread markets"
   - ❌ "Have the ml-engineer do the ML stuff"

2. **Use for context isolation:**
   - Complex tasks benefit from subagent's separate context window
   - Prevents main conversation from getting polluted

3. **Chain agents for complex tasks:**
   ```
   > Have the research-analyst find the best odds API
   > Based on that, have the data-engineer build the integration
   > Have the test-writer write tests for the new pipeline
   > Have the code-reviewer verify the implementation
   ```

4. **Use code-reviewer before commits:**
   ```
   > Have the code-reviewer check my changes to src/scoring.py
   ```

## Project-Specific Context

All agents are pre-configured with knowledge of:
- The NBA Value Betting Platform architecture
- Both Value Score algorithms (A and B)
- Database schema and data flow
- API structure and endpoints
- Freemium feature gating
- Key metrics (Brier, CLV, ROI)

## Adding New Agents

Create a new `.md` file in `.claude/agents/` with:

```markdown
---
name: your-agent-name
description: Describe when Claude should use this agent
tools: List, Of, Tools
---

You are a [role] specializing in [domain].

## Core Expertise
- Skill 1
- Skill 2

## Project Context
[Relevant project details]

## Quality Checklist
- [ ] Check 1
- [ ] Check 2
```

## Troubleshooting

**Agent not appearing in `/agents`:**
- Ensure file is in `.claude/agents/` directory
- Check YAML frontmatter syntax
- Restart Claude Code session

**Agent not being invoked:**
- Try explicit invocation: "Use the X agent to..."
- Check that description matches the task

**Wrong agent being used:**
- Make descriptions more specific
- Use explicit invocation for precision
