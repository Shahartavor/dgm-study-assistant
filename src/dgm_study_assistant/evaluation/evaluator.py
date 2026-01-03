"""RAG evaluator built on RAGAS metrics."""

from typing import Dict, List, Any, Optional
import pandas as pd

from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


class RAGEvaluator:
    """Evaluate a RAG chain using a small set of RAGAS metrics."""
    
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


    def _run_rag(self, question: str) -> Dict[str, Any]:
        result = self.rag_chain.invoke({"question": question})
        generated_answer = result["answer"]
        retrieved_docs = result["docs"]
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        return {
            "generated_answer": generated_answer,
            "retrieved_contexts": retrieved_contexts,
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


    def prepare_evaluation_data_from_dataframe(
        self,
        df: pd.DataFrame,
        question_col: str = "user_input",
        reference_answer_col: str = "reference",
        reference_contexts_col: str = "reference_contexts",
        max_samples: Optional[int] = None,
    ) -> Optional[EvaluationDataset]:
        """Prepare an EvaluationDataset by running the RAG chain over questions.

        The dataframe must contain question, reference answer, and reference
        contexts columns, whose names can be customized via the parameters.
        """
        try:
            if max_samples is not None:
                df = df.head(max_samples)

            evaluation_data: List[Dict[str, Any]] = []
            for idx, row in df.iterrows():
                question = row[question_col]
                reference_answer = row.get(reference_answer_col, "")
                reference_contexts = row.get(reference_contexts_col, [])

                print(f"Processing question {idx + 1}/{len(df)}: {str(question)[:60]}...")

                try:
                    rag_out = self._run_rag(str(question))
                    evaluation_data.append({
                        "user_input": str(question),
                        "response": rag_out["generated_answer"],
                        "reference": reference_answer if reference_answer is not None else "",
                        "retrieved_contexts": rag_out["retrieved_contexts"],
                        "reference_contexts": reference_contexts if reference_contexts is not None else [],
                    })
                except Exception as e:
                    print(f"Error processing question {idx + 1}: {e}")
                    continue

            eval_df = pd.DataFrame(evaluation_data)
            dataset = EvaluationDataset.from_pandas(eval_df)
            print(f"Prepared {len(evaluation_data)} samples for evaluation")
            return dataset
        except Exception as e:
            print(f"Error preparing evaluation data from dataframe: {e}")
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
            
            # Apply nest_asyncio only when evaluation is actually needed
            import nest_asyncio
            nest_asyncio.apply()
            
            # Run evaluation with all metrics
            results = evaluate(
                dataset=evaluation_dataset,
                metrics=list(self.metrics.values()),
                llm=self.judge_llm,
                embeddings=self.embedding_model,
                raise_exceptions=False,
            )
            
            # Convert results to pandas for analysis
            results_df = results.to_pandas()

            # Ensure metrics are numeric; RAGAS can emit NaNs when a metric fails.
            for col in ["context_recall", "faithfulness", "answer_relevancy", "context_precision"]:
                if col in results_df.columns:
                    results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
            
            # Calculate summary statistics
            summary = {
                'overall_results': {
                    'context_recall': {
                        'mean': results_df['context_recall'].dropna().mean(),
                        'std': results_df['context_recall'].dropna().std(),
                        'min': results_df['context_recall'].dropna().min(),
                        'max': results_df['context_recall'].dropna().max()
                    },
                    'faithfulness': {
                        'mean': results_df['faithfulness'].dropna().mean(),
                        'std': results_df['faithfulness'].dropna().std(),
                        'min': results_df['faithfulness'].dropna().min(),
                        'max': results_df['faithfulness'].dropna().max()
                    },
                    'answer_relevancy': {
                        'mean': results_df['answer_relevancy'].dropna().mean(),
                        'std': results_df['answer_relevancy'].dropna().std(),
                        'min': results_df['answer_relevancy'].dropna().min(),
                        'max': results_df['answer_relevancy'].dropna().max()
                    },
                    'context_precision': {
                        'mean': results_df['context_precision'].dropna().mean(),
                        'std': results_df['context_precision'].dropna().std(),
                        'min': results_df['context_precision'].dropna().min(),
                        'max': results_df['context_precision'].dropna().max()
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
    
    def evaluate_single_sample(
        self,
        question: str,
        reference_answer: Optional[str] = None,
        reference_contexts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question through the RAG pipeline.
        
        Args:
            question: Question to evaluate
            reference_answer: Optional reference answer for comparison
            
        Returns:
            Dictionary with evaluation metrics for this sample
        """
        try:
            rag_out = self._run_rag(question)
            generated_answer = rag_out["generated_answer"]
            contexts = rag_out["retrieved_contexts"]

            # Select only metrics that are supported by available fields.
            selected_metrics = [self.metrics["faithfulness"], self.metrics["answer_relevancy"]]
            if reference_answer:
                selected_metrics.append(self.metrics["context_recall"])
            if reference_contexts is not None:
                selected_metrics.append(self.metrics["context_precision"])
            
            # Create evaluation dataset for single sample
            eval_data = pd.DataFrame([{
                'user_input': question,
                'response': generated_answer,
                'reference': reference_answer or "",
                'retrieved_contexts': contexts,
                'reference_contexts': reference_contexts if reference_contexts is not None else []
            }])
            
            dataset = EvaluationDataset.from_pandas(eval_data)
            
            # Apply nest_asyncio only when evaluation is actually needed
            import nest_asyncio
            nest_asyncio.apply()
            
            # Evaluate
            results = evaluate(
                dataset=dataset,
                metrics=selected_metrics,
                llm=self.judge_llm,
                embeddings=self.embedding_model,
                raise_exceptions=False,
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
            print("\nOVERALL PERFORMANCE")
            print(f"Overall Score: {results['overall_score']:.3f}")
            print(f"Total Samples: {results['sample_count']}")
            
            print("\nRAGAS METRICS")
            for metric_name, metric_data in results['overall_results'].items():
                print(f"{metric_name.replace('_', ' ').title():15s}: {metric_data['mean']:.3f} +/- {metric_data['std']:.3f}")
            
            print("\n" + "="*60)
        else:
            print("No evaluation results available.")
