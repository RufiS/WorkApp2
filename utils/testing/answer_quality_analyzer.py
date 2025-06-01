"""Answer Quality Analyzer for user-focused debugging.

Analyzes the complete question → retrieval → answer pipeline to identify gaps
between current LLM outputs and ideal complete answers that help users.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AnswerQualityAnalyzer:
    """Analyzes answer completeness and quality from user perspective."""

    def __init__(self, retrieval_system, llm_service=None):
        """Initialize the analyzer with bulletproof LLM service discovery.

        Args:
            retrieval_system: UnifiedRetrievalSystem instance
            llm_service: LLM service for answer generation (auto-discovered if None or wrong type)
        """
        self.retrieval_system = retrieval_system
        self.logger = logger
        # Guarantee LLM service is available - core mission-critical functionality
        self.llm_service = self._discover_llm_service(llm_service)

    def _discover_llm_service(self, passed_service=None):
        """Bulletproof LLM service discovery - LLM analysis can never fail.

        Args:
            passed_service: Service passed to constructor (may be wrong type)

        Returns:
            Working LLM service instance

        Raises:
            RuntimeError: If no LLM service can be found (should never happen)
        """
        # Strategy 1: Check if passed service is actually an LLM service
        if passed_service and hasattr(passed_service, 'generate_answer'):
            try:
                # Test that it's callable
                if callable(getattr(passed_service, 'generate_answer')):
                    self.logger.info("Using passed LLM service")
                    return passed_service
            except Exception:
                pass

        # Strategy 2: Try to get LLM service from orchestrator
        try:
            from core.services.app_orchestrator import AppOrchestrator
            orchestrator = AppOrchestrator()
            if hasattr(orchestrator, 'get_llm_service'):
                llm_service = orchestrator.get_llm_service()
                if llm_service and hasattr(llm_service, 'generate_answer'):
                    self.logger.info("Using orchestrator LLM service")
                    return llm_service
        except Exception as e:
            self.logger.debug(f"Orchestrator LLM service not available: {e}")

        # Strategy 3: Try to import global LLM service
        try:
            from llm.services.llm_service import llm_service
            if llm_service and hasattr(llm_service, 'generate_answer'):
                self.logger.info("Using global LLM service instance")
                return llm_service
        except Exception as e:
            self.logger.debug(f"Global LLM service not available: {e}")

        # Strategy 4: Create new LLM service instance
        try:
            from llm.services.llm_service import LLMService
            from core.config import model_config

            # Check if we have API key
            api_key = getattr(model_config, 'api_key', None)
            if not api_key:
                # Try to get from environment
                import os
                api_key = os.getenv('OPENAI_API_KEY')

            if api_key:
                llm_service = LLMService(api_key)
                self.logger.info("Created new LLM service instance")
                return llm_service
        except Exception as e:
            self.logger.debug(f"Cannot create new LLM service: {e}")

        # Strategy 5: Try to get from Streamlit session state (if running in Streamlit)
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and 'llm_service' in st.session_state:
                llm_service = st.session_state.llm_service
                if llm_service and hasattr(llm_service, 'generate_answer'):
                    self.logger.info("Using Streamlit session LLM service")
                    return llm_service
        except Exception as e:
            self.logger.debug(f"Streamlit LLM service not available: {e}")

        # CRITICAL FAILURE - This should never happen in a working application
        error_msg = (
            "CRITICAL: Cannot find or create LLM service. "
            "LLM functionality is mission-critical and cannot fail. "
            "Check: 1) API key configuration, 2) Service initialization, 3) Import paths"
        )
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)

    def analyze_answer_completeness(
        self,
        query: str,
        expected_chunks: List[int] = None,
        expected_content_areas: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze complete answer pipeline for user value.

        Args:
            query: User question to analyze
            expected_chunks: List of chunk IDs that should be included
            expected_content_areas: List of content areas that should be covered

        Returns:
            Dictionary with comprehensive analysis
        """
        self.logger.info(f"Analyzing answer completeness for: '{query}'")

        analysis = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "expected_chunks": expected_chunks or [],
            "expected_content_areas": expected_content_areas or [],
            "retrieval_analysis": {},
            "answer_analysis": {},
            "gap_analysis": {},
            "user_impact": {}
        }

        try:
            # Step 1: Analyze current retrieval
            retrieval_result = self._analyze_retrieval(query)
            analysis["retrieval_analysis"] = retrieval_result

            # Step 2: Generate and analyze current answer
            answer_result = self._analyze_answer_generation(query, retrieval_result)
            analysis["answer_analysis"] = answer_result

            # Step 3: Identify content gaps
            gap_result = self._analyze_content_gaps(
                retrieval_result, expected_chunks, expected_content_areas
            )
            analysis["gap_analysis"] = gap_result

            # Step 4: Assess user impact
            user_impact = self._analyze_user_impact(answer_result, gap_result)
            analysis["user_impact"] = user_impact

            # Step 5: Generate actionable recommendations
            recommendations = self._generate_recommendations(analysis)
            analysis["recommendations"] = recommendations

        except Exception as e:
            self.logger.error(f"Error in answer quality analysis: {str(e)}")
            analysis["error"] = str(e)

        return analysis

    def _analyze_retrieval(self, query: str) -> Dict[str, Any]:
        """Analyze current retrieval performance."""
        try:
            # Get current retrieval results
            retrieval_results = self.retrieval_system.retrieve(query)
            context, retrieval_time, chunk_count, similarity_scores = retrieval_results

            # Get document processor for chunk analysis
            doc_processor = self.retrieval_system.document_processor

            retrieval_info = {
                "context_retrieved": context,
                "context_length": len(context) if context else 0,
                "chunk_count": chunk_count,
                "retrieval_time": retrieval_time,
                "similarity_scores": similarity_scores,
                "retrieved_chunk_ids": self._extract_chunk_ids(context),
                "chunk_analysis": {}
            }

            # Analyze individual chunks if available
            if hasattr(doc_processor, 'texts') and doc_processor.texts:
                retrieval_info["total_chunks_available"] = len(doc_processor.texts)
                retrieval_info["chunk_analysis"] = self._analyze_all_chunks(
                    query, doc_processor.texts
                )

            return retrieval_info

        except Exception as e:
            self.logger.error(f"Error analyzing retrieval: {str(e)}")
            return {"error": str(e)}

    def _analyze_answer_generation(self, query: str, retrieval_result: Dict) -> Dict[str, Any]:
        """Analyze current LLM answer generation."""
        try:
            context = retrieval_result.get("context_retrieved", "")

            if not context:
                return {
                    "answer": "No context available for answer generation",
                    "answer_length": 0,
                    "completeness_score": 0.0,
                    "actionability_score": 0.0,
                    "error": "No retrieval context"
                }

            # Generate answer using current pipeline
            answer_result = self.llm_service.generate_answer(query, context)
            answer = answer_result.get('content', '') if isinstance(answer_result, dict) else str(answer_result)

            answer_info = {
                "answer": answer,
                "answer_length": len(answer) if answer else 0,
                "completeness_analysis": self._analyze_answer_completeness_content(answer),
                "actionability_analysis": self._analyze_answer_actionability(answer),
                "user_value_assessment": self._assess_user_value(query, answer)
            }

            return answer_info

        except Exception as e:
            self.logger.error(f"Error analyzing answer generation: {str(e)}")
            return {"error": str(e)}

    def _analyze_content_gaps(
        self,
        retrieval_result: Dict,
        expected_chunks: List[int],
        expected_content_areas: List[str]
    ) -> Dict[str, Any]:
        """Analyze gaps between current and expected content."""
        retrieved_chunk_ids = retrieval_result.get("retrieved_chunk_ids", [])

        gap_analysis = {
            "missing_chunks": [],
            "missing_content_areas": [],
            "coverage_percentage": 0.0,
            "critical_gaps": []
        }

        # Analyze missing chunks
        if expected_chunks:
            missing_chunks = [cid for cid in expected_chunks if cid not in retrieved_chunk_ids]
            gap_analysis["missing_chunks"] = missing_chunks
            gap_analysis["coverage_percentage"] = (
                (len(expected_chunks) - len(missing_chunks)) / len(expected_chunks) * 100
                if expected_chunks else 0.0
            )

        # Analyze content coverage
        context = retrieval_result.get("context_retrieved", "")
        if expected_content_areas and context:
            missing_areas = []
            for area in expected_content_areas:
                if area.lower() not in context.lower():
                    missing_areas.append(area)
            gap_analysis["missing_content_areas"] = missing_areas

        # Identify critical gaps (high-impact missing content)
        gap_analysis["critical_gaps"] = self._identify_critical_gaps(
            gap_analysis["missing_chunks"],
            gap_analysis["missing_content_areas"]
        )

        return gap_analysis

    def _analyze_user_impact(self, answer_result: Dict, gap_result: Dict) -> Dict[str, Any]:
        """Analyze impact on user's ability to complete their task."""
        user_impact = {
            "can_complete_task": False,
            "completion_confidence": 0.0,
            "user_frustration_risk": "high",
            "required_followup": [],
            "task_success_probability": 0.0
        }

        answer = answer_result.get("answer", "")
        missing_chunks = gap_result.get("missing_chunks", [])

        # Assess task completion capability
        if answer and len(answer) > 100:  # Basic answer exists
            if not missing_chunks:  # All expected content retrieved
                user_impact["can_complete_task"] = True
                user_impact["completion_confidence"] = 0.9
                user_impact["user_frustration_risk"] = "low"
                user_impact["task_success_probability"] = 0.85
            elif len(missing_chunks) <= 1:  # Minor gaps
                user_impact["can_complete_task"] = True
                user_impact["completion_confidence"] = 0.7
                user_impact["user_frustration_risk"] = "medium"
                user_impact["task_success_probability"] = 0.65
                user_impact["required_followup"] = ["Additional research needed"]
            else:  # Major gaps
                user_impact["can_complete_task"] = False
                user_impact["completion_confidence"] = 0.3
                user_impact["user_frustration_risk"] = "high"
                user_impact["task_success_probability"] = 0.25
                user_impact["required_followup"] = [
                    "Significant additional research required",
                    "May need to contact supervisor or colleague",
                    "Risk of incomplete task execution"
                ]
        else:
            # No meaningful answer
            user_impact["required_followup"] = [
                "Complete re-search required",
                "Manual documentation lookup needed",
                "High probability of task failure or delay"
            ]

        return user_impact

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations for improvement."""
        recommendations = []

        gap_result = analysis.get("gap_analysis", {})
        user_impact = analysis.get("user_impact", {})
        retrieval_result = analysis.get("retrieval_analysis", {})

        # Retrieval improvements
        missing_chunks = gap_result.get("missing_chunks", [])
        if missing_chunks:
            recommendations.append(
                f"🔍 Critical: Investigate why chunks {missing_chunks} are not retrieved"
            )
            recommendations.append(
                "📊 Test lower similarity thresholds (0.05-0.2) to capture missing content"
            )

        # Coverage improvements
        coverage = gap_result.get("coverage_percentage", 0)
        if coverage < 80:
            recommendations.append(
                f"📈 Coverage only {coverage:.1f}% - increase top_k or lower threshold"
            )

        # User experience improvements
        task_success = user_impact.get("task_success_probability", 0)
        if task_success < 0.7:
            recommendations.append(
                f"⚠️ User success probability only {task_success:.1%} - prioritize completeness"
            )

        # Engine-specific recommendations
        chunk_count = retrieval_result.get("chunk_count", 0)
        if chunk_count < 5:
            recommendations.append(
                "🔧 Very few chunks retrieved - check hybrid search and BM25 integration"
            )
        elif chunk_count > 50:
            recommendations.append(
                "🎯 Too many chunks may add noise - optimize relevance scoring"
            )

        return recommendations

    def _count_chunks_in_context(self, context: str) -> int:
        """Count chunks in retrieval context."""
        if not context:
            return 0
        return context.count("[") if "[" in context else 1

    def _extract_chunk_ids(self, context: str) -> List[int]:
        """Extract chunk IDs from context."""
        chunk_ids = []
        if not context:
            return chunk_ids

        # Look for patterns like "Chunk 10:" or "[10]"
        import re
        patterns = [
            r"Chunk (\d+):",
            r"\[(\d+)\]"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, context)
            for match in matches:
                try:
                    chunk_ids.append(int(match))
                except ValueError:
                    continue

        return sorted(list(set(chunk_ids)))

    def _analyze_all_chunks(self, query: str, all_texts: List[str]) -> Dict[str, Any]:
        """Analyze similarity scores for all available chunks."""
        try:
            # Get embeddings for query and all chunks using global embedding service
            from core.embeddings.embedding_service import embedding_service

            query_embedding = embedding_service.embed_query(query)
            if query_embedding is None or len(query_embedding) == 0:
                return {"error": "Could not generate query embedding"}

            # Handle the case where embed_query returns a 2D array
            if len(query_embedding.shape) == 2:
                query_embedding = query_embedding[0]  # Get first row

            chunk_scores = []
            for i, text in enumerate(all_texts):
                try:
                    # Embed the chunk text (first 500 chars for efficiency)
                    chunk_text = text[:500] if len(text) > 500 else text
                    chunk_embedding = embedding_service.embed_query(chunk_text)

                    if chunk_embedding is not None and len(chunk_embedding) > 0:
                        # Handle 2D array case
                        if len(chunk_embedding.shape) == 2:
                            chunk_embedding = chunk_embedding[0]

                        # Calculate similarity (cosine similarity)
                        import numpy as np
                        similarity = np.dot(query_embedding, chunk_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                        )
                        chunk_scores.append({
                            "chunk_id": i,
                            "similarity_score": float(similarity),
                            "text_preview": text[:200] + "..." if len(text) > 200 else text
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing chunk {i}: {str(e)}")
                    continue

            # Sort by similarity score
            chunk_scores.sort(key=lambda x: x["similarity_score"], reverse=True)

            return {
                "total_chunks_analyzed": len(chunk_scores),
                "top_10_chunks": chunk_scores[:10],
                "bottom_10_chunks": chunk_scores[-10:] if len(chunk_scores) > 10 else [],
                "score_distribution": self._calculate_score_distribution(chunk_scores)
            }

        except Exception as e:
            self.logger.error(f"Error in chunk analysis: {str(e)}")
            return {"error": str(e)}

    def _calculate_score_distribution(self, chunk_scores: List[Dict]) -> Dict:
        """Calculate distribution statistics for similarity scores."""
        if not chunk_scores:
            return {}

        scores = [cs["similarity_score"] for cs in chunk_scores]

        return {
            "min_score": min(scores),
            "max_score": max(scores),
            "mean_score": sum(scores) / len(scores),
            "median_score": sorted(scores)[len(scores) // 2],
            "scores_above_0_8": len([s for s in scores if s > 0.8]),
            "scores_above_0_5": len([s for s in scores if s > 0.5]),
            "scores_above_0_3": len([s for s in scores if s > 0.3])
        }

    def _analyze_answer_completeness_content(self, answer: str) -> Dict[str, Any]:
        """Analyze answer content for completeness indicators."""
        if not answer:
            return {"completeness_score": 0.0, "missing_elements": ["No answer generated"]}

        completeness_indicators = {
            "step_by_step": any(word in answer.lower() for word in ["step", "first", "then", "next", "finally"]),
            "specific_tools": any(word in answer.lower() for word in ["ringcentral", "freshdesk", "sms", "ticket"]),
            "actionable_instructions": any(word in answer.lower() for word in ["click", "go to", "navigate", "select", "enter"]),
            "complete_workflow": len(answer) > 300,  # Substantial answer
            "contact_info": any(word in answer.lower() for word in ["phone", "number", "contact", "call"])
        }

        completeness_score = sum(completeness_indicators.values()) / len(completeness_indicators)

        missing_elements = [
            element for element, present in completeness_indicators.items()
            if not present
        ]

        return {
            "completeness_score": completeness_score,
            "completeness_indicators": completeness_indicators,
            "missing_elements": missing_elements
        }

    def _analyze_answer_actionability(self, answer: str) -> Dict[str, Any]:
        """Analyze how actionable the answer is for users."""
        if not answer:
            return {"actionability_score": 0.0}

        actionability_indicators = {
            "clear_instructions": ":" in answer and ("." in answer or ";" in answer),
            "specific_procedures": any(word in answer.lower() for word in ["procedure", "process", "method", "way to"]),
            "tool_references": any(word in answer.lower() for word in ["use", "open", "access", "login"]),
            "sequential_steps": answer.count("\n") > 2 or any(num in answer for num in ["1.", "2.", "3."]),
            "outcome_clarity": any(word in answer.lower() for word in ["result", "outcome", "complete", "finish"])
        }

        actionability_score = sum(actionability_indicators.values()) / len(actionability_indicators)

        return {
            "actionability_score": actionability_score,
            "actionability_indicators": actionability_indicators
        }

    def _assess_user_value(self, query: str, answer: str) -> Dict[str, Any]:
        """Assess overall value to user."""
        if not answer or len(answer) < 50:
            return {
                "user_value_score": 0.0,
                "value_assessment": "No meaningful answer provided"
            }

        # Check if answer actually addresses the query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        relevance = len(query_words.intersection(answer_words)) / len(query_words)

        value_factors = {
            "addresses_query": relevance > 0.3,
            "sufficient_detail": len(answer) > 200,
            "professional_tone": not any(word in answer.lower() for word in ["i don't know", "not sure", "maybe"]),
            "specific_context": any(word in answer.lower() for word in ["dispatch", "client", "field engineer", "appointment"])
        }

        user_value_score = sum(value_factors.values()) / len(value_factors)

        if user_value_score >= 0.8:
            assessment = "High value - users can accomplish their task"
        elif user_value_score >= 0.6:
            assessment = "Moderate value - users may need additional information"
        elif user_value_score >= 0.4:
            assessment = "Low value - users will likely be frustrated"
        else:
            assessment = "Minimal value - users cannot complete their task"

        return {
            "user_value_score": user_value_score,
            "value_factors": value_factors,
            "value_assessment": assessment,
            "query_relevance": relevance
        }

    def _identify_critical_gaps(self, missing_chunks: List[int], missing_content_areas: List[str]) -> List[str]:
        """Identify critical gaps that significantly impact user success."""
        critical_gaps = []

        # Critical chunk ranges for text message handling
        critical_chunk_ranges = {
            "ringcentral_texting": list(range(10, 13)),  # Chunks 10-12
            "text_response_workflow": [56],  # Chunk 56
            "text_tickets_handling": list(range(58, 61))  # Chunks 58-60
        }

        for area, chunk_range in critical_chunk_ranges.items():
            missing_in_range = [c for c in chunk_range if c in missing_chunks]
            if missing_in_range:
                critical_gaps.append(f"Missing {area} content (chunks {missing_in_range})")

        # Critical content areas
        critical_content = ["RingCentral", "Text Response", "SMS", "Freshdesk", "ticket handling"]
        for content in critical_content:
            if content in missing_content_areas:
                critical_gaps.append(f"Missing {content} procedures")

        return critical_gaps


def run_text_message_analysis(retrieval_system, llm_service) -> Dict[str, Any]:
    """Run comprehensive analysis for text message query."""
    analyzer = AnswerQualityAnalyzer(retrieval_system, llm_service)

    return analyzer.analyze_answer_completeness(
        query="How do I respond to a text message",
        expected_chunks=[10, 11, 12, 56, 58, 59, 60],
        expected_content_areas=[
            "RingCentral Texting",
            "Text Response",
            "SMS format",
            "Text Tickets",
            "Freshdesk ticket",
            "Field Engineer contact"
        ]
    )
