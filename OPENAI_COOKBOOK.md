# OpenAI Cookbook Integration Guide

## Overview

The Nexus AGI system now includes OpenAI Cookbook-style integration, providing LLM-enhanced capabilities for reasoning, embeddings, and text generation.

## Features

### 1. OpenAICookbookIntegration Class

A comprehensive class that provides:
- **Text Generation**: Generate text using OpenAI's chat models (GPT-3.5-turbo, GPT-4, etc.)
- **Embeddings**: Convert text to embeddings for semantic similarity analysis
- **Enhanced Reasoning**: Use LLMs to enhance problem understanding and solution generation
- **Zero-shot Classification**: Classify text without training data

### 2. Graceful Degradation

The integration works in multiple modes:
- **Full Mode**: With OpenAI API key and package installed
- **Fallback Mode**: Without API key (uses mock responses)
- **No-dependency Mode**: Works even without the openai package installed

## Quick Start

### Installation

```bash
# Install the OpenAI package
pip install openai

# Set your API key (optional, works without it)
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```python
# Import and run the system
python3 README.md  # or python3 nexus_agi

# The system will automatically:
# 1. Initialize the MetaAlgorithm_NexusCore
# 2. Demonstrate the main Nexus capabilities
# 3. Run OpenAI Cookbook examples
```

### Programmatic Usage

```python
from README import OpenAICookbookIntegration

# Initialize the integration
openai = OpenAICookbookIntegration()

# Generate text
response = openai.generate_text("Explain quantum computing")

# Get embeddings
texts = ["First text", "Second text", "Third text"]
embeddings = openai.get_embeddings(texts)

# Enhance reasoning
result = openai.enhance_reasoning(
    "How to reduce carbon emissions?",
    "Climate science, policy, economics"
)

# Classify text
classification = openai.cookbook_example_classification(
    "Scientists discover new quantum computing method",
    ["Politics", "Science", "Sports", "Entertainment"]
)
```

## Cookbook Examples

The system includes four ready-to-run examples:

### Example 1: Text Similarity with Embeddings
Demonstrates semantic similarity analysis using embeddings.

### Example 2: LLM-Enhanced Problem Reasoning
Shows how to use LLMs to enhance problem understanding.

### Example 3: Zero-shot Text Classification
Classifies text into categories without training.

### Example 4: Integration with Nexus Core
Demonstrates full integration with the Nexus AGI system.

## Integration with Nexus Core

The OpenAI integration is automatically initialized in `MetaAlgorithm_NexusCore`:

```python
core = MetaAlgorithm_NexusCore()
# Access OpenAI integration via:
core.OPENAI.generate_text("Your prompt here")
core.OPENAI.get_embeddings(["text1", "text2"])
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional)

### Custom Models

```python
# Use different models
openai = OpenAICookbookIntegration(
    model="gpt-4",
    embedding_model="text-embedding-ada-002"
)
```

## API Reference

### OpenAICookbookIntegration

#### Methods

- `generate_text(prompt, max_tokens=500, temperature=0.7)`: Generate text
- `get_embeddings(texts)`: Get embeddings for text(s)
- `enhance_reasoning(problem_description, domain_knowledge)`: Enhance problem understanding
- `cookbook_example_embeddings(texts)`: Run embeddings example
- `cookbook_example_reasoning(problem)`: Run reasoning example
- `cookbook_example_classification(text, categories)`: Run classification example

## Error Handling

The integration includes comprehensive error handling:
- Falls back to mock responses if API is unavailable
- Handles missing dependencies gracefully
- Provides informative error messages

## Best Practices

1. **API Key Management**: Store API keys in environment variables, not in code
2. **Rate Limiting**: Be mindful of API rate limits when making many requests
3. **Fallback Mode**: Test your code in fallback mode to ensure it works without API access
4. **Error Handling**: Always check for errors in production code

## Examples Output

When you run the system, you'll see output like:

```
================================================================================
OPENAI COOKBOOK EXAMPLES - LLM INTEGRATION WITH NEXUS AGI
================================================================================

--------------------------------------------------------------------------------
Example 1: Text Similarity Analysis
--------------------------------------------------------------------------------
[COOKBOOK] Example: Text Similarity with Embeddings
Analyzed 4 texts
Top 3 most similar pairs:
1. Similarity: 0.892
   Text 1: Machine learning is a subset of artificial...
   Text 2: Artificial intelligence enables computers to...

[... more examples ...]
```

## Troubleshooting

### "OpenAI package not available"
- Install the package: `pip install openai`
- The system will work in fallback mode without it

### "No API key provided"
- Set the environment variable: `export OPENAI_API_KEY="your-key"`
- Or pass it directly: `OpenAICookbookIntegration(api_key="your-key")`
- The system will work in fallback mode without it

## Contributing

To extend the OpenAI integration:

1. Add new methods to the `OpenAICookbookIntegration` class
2. Include fallback behavior for when API is unavailable
3. Add cookbook examples demonstrating the new features
4. Update this documentation

## License

Same as the main Nexus AGI system.
