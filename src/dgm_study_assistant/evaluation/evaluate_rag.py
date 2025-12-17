#!/usr/bin/env python3
"""Complete RAG Evaluation using customized RAGAS framework.
Implements domain-specific synthetic data generation and evaluation metrics.

Workflow:
1. Generate synthetic DGM questions from existing vectorstore using custom prompts
2. Run RAG chain on synthetic questions to get responses and contexts
3. Evaluate using RAGAS metrics with custom DGM technical accuracy scoring
4. Provide comprehensive analysis and actionable insights
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "src"))

from src.evaluation import RAGEvaluator, SyntheticDataGenerator
from src.dgm_study_assistant.rag.loader import build_rag_chain, get_embeddings, load_vectorstore
from src.dgm_study_assistant.llm.provider import get_llm
from src.dgm_study_assistant.config import settings


def setup_llm_and_embeddings():
    """Setup LLM and embedding models based on configuration."""
    try:
        # Get main LLM for RAG chain
        llm = get_llm()
        
        # Get embedding model
        embedding_model = get_embeddings()
        
        # For evaluation, we'll use the same LLM as judge
        # In production, you might want a more powerful judge LLM
        judge_llm = llm
        
        print(f"Using LLM: {settings.llm_provider}/{settings.llm_model}")
        print(f"Using embeddings: nomic-embed-text")
        
        return llm, embedding_model, judge_llm
        
    except Exception as e:
        print(f"Error setting up models: {e}")
        print("Make sure your .env file is configured properly and models are available.")
        return None, None, None


def load_rag_system():
    """Load the existing RAG system with vectorstore."""
    try:
        print("Loading RAG system...")
        
        # Setup models
        llm, embedding_model, judge_llm = setup_llm_and_embeddings()
        if not all([llm, embedding_model, judge_llm]):
            return None, None, None, None
        
        # Build RAG chain
        rag_chain = build_rag_chain(llm)
        
        # Load vectorstore directly
        vectorstore = load_vectorstore(embedding_model, str(parent_dir / "faiss_index"))
        
        print("RAG system loaded successfully!")
        return rag_chain, vectorstore, judge_llm, embedding_model
        
    except Exception as e:
        print(f"Error loading RAG system: {e}")
        return None, None, None, None


def generate_synthetic_dataset(vectorstore, judge_llm, embedding_model, 
                             num_samples: int = 10, 
                             save_path: str = None):
    """Generate synthetic evaluation dataset."""
    try:
        print(f"\nGenerating synthetic dataset with {num_samples} samples...")
        
        # Initialize synthetic data generator
        generator = SyntheticDataGenerator(judge_llm, embedding_model)
        
        # Load documents from vectorstore  
        documents = generator.load_documents_from_vectorstore(vectorstore, num_docs=50)
        
        if not documents:
            print("No documents loaded from vectorstore!")
            return None
        
        # Generate synthetic dataset
        synthetic_dataset = generator.generate_synthetic_dataset(
            documents=documents,
            testset_size=num_samples,
            use_specific_synthesizer=True  # Using domain-specific synthesizer
        )
        
        if synthetic_dataset is None:
            print("Failed to generate synthetic dataset!")
            return None
        
        # Save if requested
        if save_path:
            generator.save_synthetic_dataset(synthetic_dataset, save_path)
        
        return synthetic_dataset
        
    except Exception as e:
        print(f"Error generating synthetic dataset: {e}")
        return None


def run_ragas_evaluation(rag_chain, synthetic_dataset, judge_llm, embedding_model,
                        save_results_path: str = None):
    """Run RAGAS evaluation on the RAG system."""
    try:
        print("\nInitializing RAGAS evaluator...")
        
        # Initialize evaluator
        evaluator = RAGEvaluator(judge_llm, embedding_model, rag_chain)
        
        # Prepare evaluation data (run RAG on synthetic questions)
        print("Preparing evaluation data...")
        evaluation_dataset = evaluator.prepare_evaluation_data(synthetic_dataset)
        
        if evaluation_dataset is None:
            print("Failed to prepare evaluation dataset!")
            return None
        
        # Run evaluation
        print("Running RAGAS evaluation...")
        results = evaluator.evaluate_dataset(evaluation_dataset)
        
        if results is None:
            print("Evaluation failed!")
            return None
        
        # Print summary
        evaluator.print_evaluation_summary(results)
        
        # Save results if requested
        if save_results_path:
            evaluator.save_evaluation_results(results, save_results_path)
        
        return results
        
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        return None


def evaluate_single_question(rag_chain, judge_llm, embedding_model, question: str):
    """Evaluate a single question for quick testing."""
    try:
        print(f"\nEvaluating single question: {question}")
        
        evaluator = RAGEvaluator(judge_llm, embedding_model, rag_chain)
        result = evaluator.evaluate_single_sample(question)
        
        if result:
            print("\nResults:")
            for metric, value in result.items():
                if metric not in ['generated_answer', 'retrieved_contexts']:
                    print(f"{metric}: {value:.3f}" if isinstance(value, float) else f"{metric}: {value}")
            
            print(f"\nGenerated Answer: {result.get('generated_answer', 'N/A')}")
        
        return result
        
    except Exception as e:
        print(f"Error evaluating single question: {e}")
        return None


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate DGM Study Assistant RAG system using RAGAS")
    
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=10,
        help="Number of synthetic samples to generate (default: 10)"
    )
    
    parser.add_argument(
        '--save-synthetic',
        type=str,
        help="Path to save synthetic dataset JSON file"
    )
    
    parser.add_argument(
        '--save-results',
        type=str,
        default='evaluation_results.json',
        help="Path to save evaluation results (default: evaluation_results.json)"
    )
    
    parser.add_argument(
        '--question', '-q',
        type=str,
        help="Evaluate a single question instead of generating synthetic dataset"
    )
    
    parser.add_argument(
        '--load-synthetic',
        type=str,
        help="Load existing synthetic dataset instead of generating new one"
    )
    
    args = parser.parse_args()
    
    print("🤖 DGM Study Assistant - RAGAS Evaluation")
    print("=" * 50)
    print("Custom RAGAS implementation with domain-specific metrics")
    print()
    
    # Load RAG system
    rag_chain, vectorstore, judge_llm, embedding_model = load_rag_system()
    
    if not all([rag_chain, judge_llm, embedding_model]):
        print("❌ Failed to load RAG system. Exiting.")
        return
    
    # Single question evaluation
    if args.question:
        evaluate_single_question(rag_chain, judge_llm, embedding_model, args.question)
        return
    
    # Load or generate synthetic dataset
    if args.load_synthetic and Path(args.load_synthetic).exists():
        print(f"Loading synthetic dataset from {args.load_synthetic}")
        try:
            import pandas as pd
            from ragas import EvaluationDataset
            
            df = pd.read_json(args.load_synthetic)
            synthetic_dataset = EvaluationDataset.from_pandas(df)
            print(f"Loaded {len(synthetic_dataset)} synthetic samples")
        except Exception as e:
            print(f"Error loading synthetic dataset: {e}")
            return
    else:
        if not vectorstore:
            print("❌ Vectorstore required for synthetic data generation. Exiting.")
            return
            
        synthetic_dataset = generate_synthetic_dataset(
            vectorstore, 
            judge_llm, 
            embedding_model,
            num_samples=args.samples,
            save_path=args.save_synthetic
        )
        
        if synthetic_dataset is None:
            print("❌ Failed to generate synthetic dataset. Exiting.")
            return
    
    # Run RAGAS evaluation
    results = run_ragas_evaluation(
        rag_chain,
        synthetic_dataset, 
        judge_llm,
        embedding_model,
        save_results_path=args.save_results
    )
    
    if results:
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Results saved to: {args.save_results}")
        
        # Provide recommendations based on results
        overall_score = results.get('overall_score', 0)
        if overall_score >= 0.8:
            print("\n🎉 Excellent RAG performance!")
        elif overall_score >= 0.6:
            print("\n👍 Good RAG performance with room for improvement.")
        else:
            print("\n⚠️  RAG performance needs improvement. Consider:")
            print("   - Improving document chunking strategy")
            print("   - Tuning retrieval parameters")
            print("   - Using a more powerful LLM")
            print("   - Improving prompt engineering")
    else:
        print("❌ Evaluation failed.")


if __name__ == "__main__":
    main()
