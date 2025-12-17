# RAG Evaluation with Custom RAGAS Implementation

This dedicated evaluation folder provides comprehensive RAG system evaluation using customized RAGAS framework with domain-specific prompts and metrics for Deep Generative Models.

## 📁 Folder Structure

```
evaluation/
├── src/
│   ├── __init__.py              # Package exports
│   ├── synthetic_generator.py   # Custom RAGAS synthetic data generation
│   └── evaluator.py            # Main RAGAS evaluation with custom metrics
├── evaluate_rag.py             # Main evaluation script
├── requirements.txt            # Evaluation dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd evaluation
pip install -r requirements.txt
```

### 2. Basic Evaluation (10 samples)
```bash
python evaluate_rag.py --samples 10
```

### 3. Full Evaluation (20 samples with saved results)
```bash
python evaluate_rag.py --samples 20 --save-results my_evaluation.json
```

### 4. Test Single Question
```bash
python evaluate_rag.py --question "What is the ELBO in variational autoencoders?"
```

## 📋 Detailed Usage

### Complete Evaluation Workflow
```bash
# Run evaluation with synthetic data generation
python evaluate_rag.py --samples 15 --save-synthetic synthetic_data.json --save-results results.json

# Reuse existing synthetic dataset (faster)
python evaluate_rag.py --load-synthetic synthetic_data.json --save-results new_results.json
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--samples`, `-n` | Number of synthetic samples to generate | 10 |
| `--question`, `-q` | Evaluate single question instead of full dataset | None |
| `--save-synthetic` | Save synthetic dataset to JSON file | None |
| `--load-synthetic` | Load existing synthetic dataset | None |
| `--save-results` | Save evaluation results | `evaluation_results.json` |

## 🔧 Prerequisites

1. **RAG System Setup**: Ensure your main RAG system is working and has a built FAISS index at `../faiss_index/`
2. **Environment Configuration**: Your `.env` file should be configured with LLM settings
3. **Dependencies**: Run `pip install -r requirements.txt` in this directory

## 📊 Evaluation Metrics

The system evaluates 5 metrics:

### Core RAGAS Metrics
- **Context Recall** (20% weight): How well retrieved contexts support reference answers
- **Faithfulness** (25% weight): How well generated answers align with retrieved contexts  
- **Answer Relevancy** (25% weight): How relevant generated answers are to questions
- **Context Precision** (10% weight): Precision of retrieved contexts

### Custom Metric
- **DGM Technical Accuracy** (20% weight): Custom metric evaluating technical correctness of Deep Generative Models concepts

### Overall Score
Weighted combination: `0.20×context_recall + 0.25×faithfulness + 0.25×answer_relevancy + 0.10×context_precision + 0.20×dgm_technical_accuracy`

## 🔍 Features

### Custom Synthetic Data Generation
- Domain-specific prompts for Deep Generative Models
- Few-shot examples for VAEs, GANs, and Diffusion Models
- Technical question generation focused on mathematical formulations

### Structured Evaluation
- LLM-as-a-Judge with JSON structured responses
- Custom scoring criteria for technical accuracy
- Comprehensive statistical analysis

## 📈 Example Output

```
🤖 DGM Study Assistant - RAGAS Evaluation
==================================================
Custom RAGAS implementation with domain-specific metrics

Using LLM: ollama/granite4:micro
Using embeddings: nomic-embed-text
Loading RAG system...

Generating synthetic dataset with 10 samples...
Loading 50 documents for DGM-focused synthetic generation
Successfully generated 10 synthetic samples

Running RAGAS evaluation...
Evaluation completed!
Overall Score: 0.742
Context Recall: 0.850
Faithfulness: 0.721
Answer Relevancy: 0.698
Context Precision: 0.702
DGM Technical Accuracy: 0.680

✅ Evaluation completed successfully!
Results saved to: evaluation_results.json
👍 Good RAG performance with room for improvement.
```

## 🔧 Troubleshooting

### Common Issues

**FAISS index not found**: 
```bash
# Make sure to build the RAG system first from the main directory
cd ..
python -c "from src.dgm_study_assistant.rag.loader import build_rag_chain, get_embeddings; from src.dgm_study_assistant.llm.provider import get_llm; build_rag_chain(get_llm())"
```

**Import errors**:
```bash
# Make sure you're in the evaluation directory
cd evaluation
python evaluate_rag.py --samples 5
```

**Memory issues**: Reduce `--samples` parameter for large datasets

**LLM not responding**: Check that your LLM is running and accessible via the configured provider

## 🎯 Performance Interpretation

- **Score ≥ 0.8**: Excellent RAG performance
- **Score ≥ 0.6**: Good performance with room for improvement  
- **Score < 0.6**: Consider improving:
  - Document chunking strategy
  - Retrieval parameters (k, similarity threshold)
  - LLM model choice
  - Prompt engineering

## 🔄 Workflow Integration

This evaluation can be integrated into your development workflow:

```bash
# After making changes to your RAG system
cd evaluation
python evaluate_rag.py --samples 20 --save-results "experiment_$(date +%Y%m%d).json"
```

## 📚 Technical Implementation

The evaluation follows patterns from advanced RAG evaluation notebooks:
- Custom prompt engineering for domain-specific synthetic generation
- Structured output evaluation with JSON validation
- Multi-metric scoring with weighted combinations
- Statistical analysis with mean, std, min, max reporting
