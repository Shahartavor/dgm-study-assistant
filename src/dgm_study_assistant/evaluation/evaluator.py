"""
RAG Evaluator using RAGAS framework with custom metrics.
Implements structured output evaluation and domain-specific scoring.
"""

import asyncio
import nest_asyncio
from typing import Dict, List, Any, Optional
import pandas as pd

from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Apply nest_asyncio to handle nested async loops
nest_asyncio.apply()
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


class RAGEvaluator:
    """
    Main RAG evaluator using RAGAS metrics with custom scoring.
    Implements structured evaluation with domain-specific criteria.
    """
    
    def __init__(self, judge_llm, embedding_model, rag_chain):
        """
        Initialize RAG evaluator with RAGAS metrics.
        
        Args:
            judge_llm: LLM to use as judge for evaluation metrics
            embedding_model: Embedding model for context evaluation
            rag_chain: The RAG pipeline to evaluate
        """
        self.rag_chain = rag_chain
        
        # Wrap LLMs for RAGAS
        self.judge_llm = LangchainLLMWrapper(judge_llm)
        self.embedding_model = LangchainEmbeddingsWrapper(embedding_model)
        
        # Initialize RAGAS metrics 
        self.metrics = {
            'context_recall': LLMContextRecall(llm=self.judge_llm),
            'faithfulness': Faithfulness(llm=self.judge_llm),
            'answer_relevancy': AnswerRelevancy(
                llm=self.judge_llm, 
                embeddings=self.embedding_model
            ),
            'context_precision': ContextPrecision(llm=self.judge_llm)
        }
    
    
    def prepare_evaluation_data(self, synthetic_dataset) -> EvaluationDataset:
        """
        Convert synthetic dataset to format suitable for RAG evaluation.
        
        Args:
            synthetic_dataset: RAGAS generated synthetic dataset
            
        Returns:
            EvaluationDataset ready for evaluation
        """
        try:
            # Get questions from synthetic dataset
            df = synthetic_dataset.to_pandas()
            
            # Run RAG chain on each question to get responses and contexts
            evaluation_data = []
            
            for idx, row in df.iterrows():
                question = row['user_input']
                reference_answer = row['reference']
                reference_contexts = row['reference_contexts']
                
                print(f"Processing question {idx + 1}/{len(df)}: {question[:60]}...")
                
                try:
                    # Run RAG chain
                    rag_result = self.rag_chain.invoke({"question": question})
                    generated_answer = rag_result["answer"]
                    retrieved_docs = rag_result["docs"]
                    
                    # Extract contexts from retrieved documents
                    retrieved_contexts = [doc.page_content for doc in retrieved_docs]
                    
                    evaluation_data.append({
                        'user_input': question,
                        'response': generated_answer,
                        'reference': reference_answer,
                        'retrieved_contexts': retrieved_contexts,
                        'reference_contexts': reference_contexts
                    })
                    
                except Exception as e:
                    print(f"Error processing question {idx + 1}: {e}")
                    continue
            
            # Convert to RAGAS EvaluationDataset
            eval_df = pd.DataFrame(evaluation_data)
            dataset = EvaluationDataset.from_pandas(eval_df)
            
            print(f"Prepared {len(evaluation_data)} samples for evaluation")
            return dataset
            
        except Exception as e:
            print(f"Error preparing evaluation data: {e}")
            return None
    
    def evaluate_dataset(self, evaluation_dataset: EvaluationDataset) -> Dict[str, Any]:
        """
        Evaluate RAG system using RAGAS metrics.
        
        Args:
            evaluation_dataset: Dataset prepared for evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            print("Starting RAGAS evaluation...")
            
            # Run evaluation with all metrics
            results = evaluate(
                dataset=evaluation_dataset,
                metrics=list(self.metrics.values()),
                llm=self.judge_llm,
                embeddings=self.embedding_model,
                raise_exceptions=False,
                is_async=False
            )
            
            # Convert results to pandas for analysis
            results_df = results.to_pandas()
            
            # Calculate summary statistics
            summary = {
                'overall_results': {
                    'context_recall': {
                        'mean': results_df['context_recall'].mean(),
                        'std': results_df['context_recall'].std(),
                        'min': results_df['context_recall'].min(),
                        'max': results_df['context_recall'].max()
                    },
                    'faithfulness': {
                        'mean': results_df['faithfulness'].mean(),
                        'std': results_df['faithfulness'].std(),
                        'min': results_df['faithfulness'].min(),
                        'max': results_df['faithfulness'].max()
                    },
                    'answer_relevancy': {
                        'mean': results_df['answer_relevancy'].mean(),
                        'std': results_df['answer_relevancy'].std(),
                        'min': results_df['answer_relevancy'].min(),
                        'max': results_df['answer_relevancy'].max()
                    },
                    'context_precision': {
                        'mean': results_df['context_precision'].mean(),
                        'std': results_df['context_precision'].std(),
                        'min': results_df['context_precision'].min(),
                        'max': results_df['context_precision'].max()
                    }
                },
                'sample_count': len(results_df),
                'detailed_results': results_df.to_dict('records')
            }
            
            # Calculate overall score (weighted average of core RAGAS metrics)
            overall_score = (
                summary['overall_results']['context_recall']['mean'] * 0.25 +
                summary['overall_results']['faithfulness']['mean'] * 0.30 +
                summary['overall_results']['answer_relevancy']['mean'] * 0.30 +
                summary['overall_results']['context_precision']['mean'] * 0.15
            )
            
            summary['overall_score'] = overall_score
            
            print(f"Evaluation completed!")
            print(f"Overall Score: {overall_score:.3f}")
            print(f"Context Recall: {summary['overall_results']['context_recall']['mean']:.3f}")
            print(f"Faithfulness: {summary['overall_results']['faithfulness']['mean']:.3f}")
            print(f"Answer Relevancy: {summary['overall_results']['answer_relevancy']['mean']:.3f}")
            print(f"Context Precision: {summary['overall_results']['context_precision']['mean']:.3f}")
            
            return summary
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None
    
    def evaluate_single_sample(self, question: str, reference_answer: str = None) -> Dict[str, Any]:
        """
        Evaluate a single question through the RAG pipeline.
        
        Args:
            question: Question to evaluate
            reference_answer: Optional reference answer for comparison
            
        Returns:
            Dictionary with evaluation metrics for this sample
        """
        try:
            # Run RAG chain
            result = self.rag_chain.invoke({"question": question})
            generated_answer = result["answer"]
            retrieved_docs = result["docs"]
            
            # Extract contexts
            contexts = [doc.page_content for doc in retrieved_docs]
            
            # Create evaluation dataset for single sample
            eval_data = pd.DataFrame([{
                'user_input': question,
                'response': generated_answer,
                'reference': reference_answer or "",
                'retrieved_contexts': contexts,
                'reference_contexts': contexts  # Use retrieved contexts as reference
            }])
            
            dataset = EvaluationDataset.from_pandas(eval_data)
            
            # Evaluate
            results = evaluate(
                dataset=dataset,
                metrics=list(self.metrics.values()),
                llm=self.judge_llm,
                embeddings=self.embedding_model,
                raise_exceptions=False,
                is_async=False
            )
            
            results_dict = results.to_pandas().iloc[0].to_dict()
            results_dict['generated_answer'] = generated_answer
            results_dict['retrieved_contexts'] = contexts
            
            return results_dict
            
        except Exception as e:
            print(f"Error evaluating single sample: {e}")
            return None
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to JSON file."""
        try:
            import json
            from pathlib import Path
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"Saved evaluation results to {filepath}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a human-readable summary of evaluation results."""
        print("\n" + "="*60)
        print("RAG EVALUATION RESULTS (RAGAS Metrics)")
        print("="*60)
        
        if results and 'overall_results' in results:
            print(f"\n📊 OVERALL PERFORMANCE")
            print(f"Overall Score: {results['overall_score']:.3f}")
            print(f"Total Samples: {results['sample_count']}")
            
            print(f"\n🔍 RAGAS METRICS")
            for metric_name, metric_data in results['overall_results'].items():
                print(f"{metric_name.replace('_', ' ').title():15s}: {metric_data['mean']:.3f} ± {metric_data['std']:.3f}")
            
            print("\n" + "="*60)
        else:
            print("No evaluation results available.")
