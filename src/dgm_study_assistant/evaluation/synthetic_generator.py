"""
Synthetic Data Generation using RAGAS TestsetGenerator.
Simplified implementation using core RAGAS functionality.
"""

from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


class SyntheticDataGenerator:
    """Generate synthetic evaluation datasets using RAGAS TestsetGenerator."""
    
    def __init__(self, llm, embedding_model):
        """
        Initialize synthetic data generator.
        
        Args:
            llm: LangChain LLM for generation (e.g., ChatNVIDIA or ChatOllama)
            embedding_model: LangChain embeddings for document processing
        """
        self.generator_llm = LangchainLLMWrapper(llm)
        self.embedding_model = LangchainEmbeddingsWrapper(embedding_model)
        
        # Initialize RAGAS generator
        self.generator = TestsetGenerator(
            llm=self.generator_llm,
            embedding_model=self.embedding_model,
        )
        
    
    def load_documents_from_vectorstore(self, vectorstore, num_docs: int = 50) -> List[Document]:
        """
        Load documents from existing vectorstore for synthetic generation.
        Ensures documents are long enough for RAGAS (>100 tokens).
        
        Args:
            vectorstore: FAISS vectorstore containing DGM course materials
            num_docs: Number of documents to sample for generation
            
        Returns:
            List of Document objects with sufficient length
        """
        try:
            # Sample documents from vectorstore
            sample_queries = [
                "deep generative models",
                "variational autoencoders",
                "generative adversarial networks", 
                "diffusion models",
                "expectation maximization"
            ]
            
            all_documents = []
            docs_per_query = (num_docs * 2) // len(sample_queries)  # Get more docs to filter
            
            for query in sample_queries:
                docs = vectorstore.similarity_search(query, k=docs_per_query)
                all_documents.extend(docs)
            
            # Filter for documents with sufficient length (>100 tokens, roughly >400 characters)
            long_documents = []
            short_documents = []
            
            for doc in all_documents:
                if len(doc.page_content) > 400:  # Approximate 100+ tokens
                    long_documents.append(doc)
                else:
                    short_documents.append(doc)
            
            print(f"Found {len(long_documents)} long documents, {len(short_documents)} short documents")
            
            # If we don't have enough long documents, combine short ones
            if len(long_documents) < num_docs and short_documents:
                print("Combining short documents to create longer ones...")
                combined_documents = self._combine_short_documents(short_documents, target_count=num_docs-len(long_documents))
                long_documents.extend(combined_documents)
            
            result_documents = long_documents[:num_docs]
            print(f"Loading {len(result_documents)} documents for DGM-focused synthetic generation")
            
            # Debug: show document lengths
            for i, doc in enumerate(result_documents[:3]):
                print(f"Doc {i+1} length: {len(doc.page_content)} chars")
                
            return result_documents
            
        except Exception as e:
            print(f"Error loading documents from vectorstore: {e}")
            return []
    
    def _combine_short_documents(self, short_docs: List[Document], target_count: int) -> List[Document]:
        """Combine multiple short documents into longer ones."""
        combined_docs = []
        
        # Combine documents in groups of 3-4
        for i in range(0, min(len(short_docs), target_count * 4), 4):
            group = short_docs[i:i+4]
            if len(group) >= 2:  # Need at least 2 docs to combine
                combined_content = "\n\n".join([doc.page_content for doc in group])
                combined_metadata = group[0].metadata.copy()
                combined_metadata['combined_docs'] = len(group)
                
                combined_doc = Document(
                    page_content=combined_content,
                    metadata=combined_metadata
                )
                combined_docs.append(combined_doc)
                
                if len(combined_docs) >= target_count:
                    break
        
        return combined_docs
    
    def generate_synthetic_dataset(self, 
                                 documents: List[Document], 
                                 testset_size: int = 20,
                                 use_specific_synthesizer: bool = True) -> Optional[object]:
        """
        Generate synthetic question-answer pairs from documents.
        
        Args:
            documents: List of documents to generate from
            testset_size: Number of synthetic samples to generate
            use_specific_synthesizer: Unused parameter (kept for compatibility)
            
        Returns:
            RAGAS EvaluationDataset or None if generation fails
        """
        try:
            print(f"Generating {testset_size} synthetic samples from {len(documents)} documents...")
            
            # Apply nest_asyncio only when generation is actually needed
            import nest_asyncio
            nest_asyncio.apply()
            
            # Generate synthetic dataset using default RAGAS configuration
            generated_test_set = self.generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=testset_size,
                raise_exceptions=False
            )
            
            print(f"Successfully generated {len(generated_test_set)} synthetic samples")
            return generated_test_set
            
        except Exception as e:
            print(f"Error generating synthetic dataset: {e}")
            return None
    
    def save_synthetic_dataset(self, dataset, filepath: str):
        """Save synthetic dataset to file."""
        try:
            df = dataset.to_pandas()
            df.to_json(filepath, orient='records', indent=2)
            print(f"Saved synthetic dataset to {filepath}")
        except Exception as e:
            print(f"Error saving dataset: {e}")
    
    def load_synthetic_dataset(self, filepath: str):
        """Load synthetic dataset from file."""
        try:
            import pandas as pd
            from ragas import EvaluationDataset
            
            df = pd.read_json(filepath)
            # Convert back to RAGAS format if needed
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
