"""Testing Controller for WorkApp2.

Handles UI integration for systematic engine testing and comparison.
"""

import streamlit as st
import logging
import time
from typing import Dict, Any, List, Optional

from utils.testing import TestRunner, TEST_CONFIGURATIONS
from utils.testing.engine_comparison import ComparisonResults
from utils.testing.answer_quality_analyzer import AnswerQualityAnalyzer, run_text_message_analysis

logger = logging.getLogger(__name__)


class TestingController:
    """Controller for systematic testing UI functionality."""
    
    def __init__(self, orchestrator):
        """Initialize the testing controller.
        
        Args:
            orchestrator: AppOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.logger = logger
        
        # Initialize test runner when retrieval system is available
        self._test_runner = None
    
    @property
    def test_runner(self) -> Optional[TestRunner]:
        """Get test runner instance, initializing if needed."""
        if self._test_runner is None:
            try:
                _, _, retrieval_system = self.orchestrator.get_services()
                self._test_runner = TestRunner(retrieval_system)
            except Exception as e:
                self.logger.error(f"Failed to initialize test runner: {str(e)}")
                return None
        return self._test_runner
    
    def render_testing_section(self) -> None:
        """Render the systematic testing section in the UI."""
        # Create tabs for different testing approaches
        tab1, tab2 = st.tabs(["🔧 Engine Testing", "📋 Answer Quality Analysis"])
        
        with tab1:
            self._render_engine_testing()
        
        with tab2:
            self._render_answer_quality_analysis()
    
    def _render_engine_testing(self) -> None:
        """Render the systematic engine testing section."""
        st.header("🔬 Systematic Engine Testing")
        st.markdown("""
        Test the same query across different retrieval engine configurations to identify optimal settings.
        Compare vector search, hybrid search, and reranking engines with various parameters.
        """)
        
        # Validate environment first
        validation_result = self._validate_testing_environment()
        if not validation_result["valid"]:
            self._render_validation_errors(validation_result)
            return
        
        # Show warnings if any
        if validation_result["warnings"]:
            self._render_validation_warnings(validation_result)
        
        # Test query input
        test_query = st.text_area(
            "Test Query",
            value="How do I respond to a text message",
            height=100,
            help="Enter a query to test across all engine configurations"
        )
        
        # Test suite selection
        test_suites = self.test_runner.get_available_test_suites()
        selected_suite = st.selectbox(
            "Select Test Suite",
            options=list(test_suites.keys()),
            format_func=lambda x: f"{test_suites[x]['name']} ({test_suites[x]['estimated_time']})",
            help="Choose which configurations to test"
        )
        
        # Show selected configurations
        with st.expander("Configuration Details", expanded=False):
            suite_info = test_suites[selected_suite]
            st.markdown(f"**{suite_info['name']}**")
            st.markdown(suite_info['description'])
            st.markdown(f"**Estimated Time:** {suite_info['estimated_time']}")
            st.markdown("**Configurations to test:**")
            for config_name in suite_info['configurations']:
                try:
                    config = TEST_CONFIGURATIONS.get_configuration_by_name(config_name)
                    st.markdown(f"- **{config.name}**: {config.description}")
                except ValueError:
                    st.markdown(f"- {config_name} (configuration not found)")
        
        # Test execution button
        col1, col2 = st.columns([1, 3])
        
        with col1:
            run_test = st.button(
                "🚀 Run Test",
                type="primary",
                disabled=not test_query.strip(),
                help="Run the selected test suite with the provided query"
            )
        
        with col2:
            if st.button("🔄 Clear Results", help="Clear previous test results"):
                if "test_results" in st.session_state:
                    del st.session_state["test_results"]
                st.rerun()
        
        # Run test if button clicked
        if run_test and test_query.strip():
            self._execute_test_suite(test_query.strip(), selected_suite)
        
        # Display results if available
        if "test_results" in st.session_state:
            self._render_test_results(st.session_state["test_results"])
    
    def _validate_testing_environment(self) -> Dict[str, Any]:
        """Validate that testing environment is ready."""
        if not self.test_runner:
            return {
                "valid": False,
                "issues": ["Failed to initialize test runner"],
                "warnings": []
            }
        
        return self.test_runner.validate_test_environment()
    
    def _render_validation_errors(self, validation_result: Dict[str, Any]) -> None:
        """Render validation errors in the UI."""
        st.error("❌ Testing Environment Issues")
        
        for issue in validation_result["issues"]:
            st.error(f"• {issue}")
        
        st.info("""
        **To resolve these issues:**
        1. Upload documents to create an index
        2. Ensure all services are properly initialized
        3. Check the application logs for more details
        """)
    
    def _render_validation_warnings(self, validation_result: Dict[str, Any]) -> None:
        """Render validation warnings in the UI."""
        with st.expander("⚠️ Testing Environment Warnings", expanded=False):
            for warning in validation_result["warnings"]:
                st.warning(f"• {warning}")
    
    def _execute_test_suite(self, query: str, suite_name: str) -> None:
        """Execute the selected test suite."""
        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Start testing
            status_text.text("🔄 Initializing test suite...")
            progress_bar.progress(10)
            
            # Execute the appropriate test
            if suite_name == "baseline":
                status_text.text("🔄 Running baseline comparison...")
                progress_bar.progress(30)
                results = self.test_runner.run_baseline_comparison(query)
                
            elif suite_name == "full":
                status_text.text("🔄 Running full comparison...")
                progress_bar.progress(30)
                results = self.test_runner.run_full_comparison(query)
                
            elif suite_name == "parameter_sweep":
                status_text.text("🔄 Running parameter sweep...")
                progress_bar.progress(30)
                results = self.test_runner.run_parameter_sweep(query)
            
            else:
                st.error(f"Unknown test suite: {suite_name}")
                return
            
            # Update progress
            progress_bar.progress(80)
            status_text.text("🔄 Analyzing results...")
            
            # Generate report
            report = self.test_runner.generate_test_report(results)
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Test completed successfully!")
            
            # Store results in session state
            st.session_state["test_results"] = {
                "results": results,
                "report": report,
                "suite_name": suite_name
            }
            
            # Clear progress indicators after a moment
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Force refresh to show results
            st.rerun()
            
        except Exception as e:
            self.logger.error(f"Error executing test suite: {str(e)}")
            st.error(f"❌ Test execution failed: {str(e)}")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
    
    def _render_test_results(self, test_data: Dict[str, Any]) -> None:
        """Render comprehensive test results."""
        results = test_data["results"]
        report = test_data["report"]
        suite_name = test_data["suite_name"]
        
        st.header("📊 Test Results")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Configurations",
                report["metadata"]["total_configurations"]
            )
        
        with col2:
            successful_count = report["summary"]["successful_configurations"]
            total_count = report["summary"]["total_configurations"]
            st.metric(
                "Success Rate",
                f"{successful_count}/{total_count}",
                f"{report['summary']['overall_success_rate']:.1%}"
            )
        
        with col3:
            if report["summary"]["best_configuration"]:
                best_score = report["summary"]["best_configuration"]["success_score"]
                st.metric("Best Score", f"{best_score:.3f}")
            else:
                st.metric("Best Score", "N/A")
        
        with col4:
            st.metric(
                "Test Duration",
                f"{report['metadata']['test_duration']:.1f}s"
            )
        
        # Performance rankings
        st.subheader("🏆 Performance Rankings")
        
        if report["summary"]["performance_rankings"]:
            ranking_data = []
            for ranking in report["summary"]["performance_rankings"]:
                ranking_data.append({
                    "Rank": ranking["rank"],
                    "Configuration": ranking["configuration"],
                    "Success": "✅" if ranking["success"] else "❌",
                    "Score": f"{ranking['success_score']:.3f}",
                    "Quality": f"{ranking['quality_score']:.3f}",
                    "Time (s)": f"{ranking['retrieval_time']:.2f}"
                })
            
            st.dataframe(
                ranking_data,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No successful configurations found in the test results.")
        
        # Insights and recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💡 Insights")
            if report["insights"]:
                for insight in report["insights"]:
                    st.markdown(f"• {insight}")
            else:
                st.info("No specific insights generated.")
        
        with col2:
            st.subheader("🎯 Recommendations")
            if report["summary"]["recommendations"]:
                for recommendation in report["summary"]["recommendations"]:
                    st.markdown(f"• {recommendation}")
            else:
                st.info("No specific recommendations available.")
        
        # Detailed results
        with st.expander("📋 Detailed Results", expanded=False):
            for result_detail in report["detailed_results"]:
                with st.container():
                    st.markdown(f"### {result_detail['rank']}. {result_detail['configuration']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Success:** {'✅' if result_detail['success'] else '❌'}")
                        st.markdown(f"**Overall Score:** {result_detail['scores']['overall']:.3f}")
                    
                    with col2:
                        st.markdown(f"**Quality Score:** {result_detail['scores']['context_quality']:.3f}")
                        st.markdown(f"**Retrieval Time:** {result_detail['performance']['retrieval_time']:.2f}s")
                    
                    with col3:
                        st.markdown(f"**Chunks:** {result_detail['performance']['chunk_count']}")
                        st.markdown(f"**Context Length:** {result_detail['performance']['context_length']}")
                    
                    if result_detail["context_preview"]:
                        st.markdown("**Context Preview:**")
                        st.text_area(
                            "Context Preview",
                            result_detail["context_preview"],
                            height=100,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"context_preview_{result_detail['rank']}"
                        )
                    
                    if result_detail["error_message"]:
                        st.error(f"Error: {result_detail['error_message']}")
                    
                    st.divider()
        
        # Next steps
        st.subheader("🚀 Next Steps")
        if report["next_steps"]:
            for step in report["next_steps"]:
                st.markdown(f"• {step}")
        else:
            st.info("Consider running additional tests with different queries.")
        
        # Export options
        st.subheader("📥 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download Report JSON"):
                st.download_button(
                    label="Download JSON Report",
                    data=str(report),
                    file_name=f"test_report_{suite_name}_{int(time.time())}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Download Rankings CSV"):
                # Convert rankings to CSV format
                csv_data = "Rank,Configuration,Success,Score,Quality,Time\n"
                for ranking in report["summary"]["performance_rankings"]:
                    csv_data += f"{ranking['rank']},{ranking['configuration']},{ranking['success']},{ranking['success_score']:.3f},{ranking['quality_score']:.3f},{ranking['retrieval_time']:.2f}\n"
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"rankings_{suite_name}_{int(time.time())}.csv",
                    mime="text/csv"
                )
    
    def _render_answer_quality_analysis(self) -> None:
        """Render the answer quality analysis section."""
        st.header("📋 Answer Quality Analysis")
        st.markdown("""
        Analyze the complete question → retrieval → answer pipeline to identify gaps between current 
        LLM outputs and ideal complete answers that help users accomplish their tasks.
        """)
        
        # Validate environment first
        validation_result = self._validate_testing_environment()
        if not validation_result["valid"]:
            self._render_validation_errors(validation_result)
            return
        
        # Quick analysis for text message example
        st.subheader("🚀 Quick Analysis: Text Message Query")
        st.markdown("""
        Run comprehensive analysis on the critical "text message" query to understand why users 
        aren't getting complete, actionable answers.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button(
                "🔍 Analyze Text Message Query",
                type="primary",
                help="Run comprehensive analysis for 'How do I respond to a text message'"
            ):
                self._execute_answer_quality_analysis("text_message")
        
        with col2:
            st.info("""
            **Expected Content Areas:**
            • RingCentral Texting (chunks 10-12)
            • Text Response workflow (chunk 56) 
            • Text Tickets handling (chunks 58-60)
            """)
        
        # Custom query analysis
        st.subheader("🎯 Custom Query Analysis")
        
        custom_query = st.text_area(
            "Enter Query to Analyze",
            value="How do I handle appointment cancellations?",
            height=100,
            help="Enter any dispatch-related question to analyze answer completeness"
        )
        
        # Expected content specification
        col1, col2 = st.columns(2)
        
        with col1:
            expected_chunks = st.text_input(
                "Expected Chunk IDs (comma-separated)",
                placeholder="e.g., 105, 106, 107",
                help="Specific chunk numbers that should be included for complete answer"
            )
        
        with col2:
            expected_content = st.text_area(
                "Expected Content Areas (one per line)",
                placeholder="Same Day Cancellation\nCancellation Policy\nReschedule Process",
                help="Content areas that should be covered for complete guidance"
            )
        
        if st.button(
            "🔬 Analyze Custom Query",
            disabled=not custom_query.strip(),
            help="Run comprehensive analysis on your custom query"
        ):
            # Parse expected inputs
            chunk_list = []
            if expected_chunks.strip():
                try:
                    chunk_list = [int(x.strip()) for x in expected_chunks.split(",") if x.strip()]
                except ValueError:
                    st.error("Invalid chunk IDs. Please enter comma-separated numbers.")
                    return
            
            content_list = []
            if expected_content.strip():
                content_list = [line.strip() for line in expected_content.split("\n") if line.strip()]
            
            self._execute_custom_answer_analysis(
                custom_query.strip(), 
                chunk_list, 
                content_list
            )
        
        # Display results if available
        if "answer_quality_results" in st.session_state:
            self._render_answer_quality_results(st.session_state["answer_quality_results"])
    
    def _execute_answer_quality_analysis(self, analysis_type: str) -> None:
        """Execute answer quality analysis."""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Initializing answer quality analyzer...")
            progress_bar.progress(20)
            
            # Get services
            llm_service, _, retrieval_system = self.orchestrator.get_services()
            
            if analysis_type == "text_message":
                status_text.text("🔄 Analyzing text message query...")
                progress_bar.progress(50)
                
                results = run_text_message_analysis(retrieval_system, llm_service)
            
            else:
                st.error(f"Unknown analysis type: {analysis_type}")
                return
            
            progress_bar.progress(90)
            status_text.text("🔄 Generating recommendations...")
            
            # Store results
            st.session_state["answer_quality_results"] = {
                "analysis": results,
                "analysis_type": analysis_type
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis completed!")
            
            # Clear progress indicators and refresh
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            st.rerun()
            
        except Exception as e:
            self.logger.error(f"Error in answer quality analysis: {str(e)}")
            st.error(f"❌ Analysis failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    def _execute_custom_answer_analysis(
        self, 
        query: str, 
        expected_chunks: List[int], 
        expected_content: List[str]
    ) -> None:
        """Execute custom answer quality analysis."""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Initializing analyzer...")
            progress_bar.progress(20)
            
            # Get services
            llm_service, _, retrieval_system = self.orchestrator.get_services()
            
            status_text.text(f"🔄 Analyzing query: {query[:50]}...")
            progress_bar.progress(50)
            
            # Run analysis
            analyzer = AnswerQualityAnalyzer(retrieval_system, llm_service)
            results = analyzer.analyze_answer_completeness(
                query=query,
                expected_chunks=expected_chunks,
                expected_content_areas=expected_content
            )
            
            progress_bar.progress(90)
            status_text.text("🔄 Generating recommendations...")
            
            # Store results
            st.session_state["answer_quality_results"] = {
                "analysis": results,
                "analysis_type": "custom"
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis completed!")
            
            # Clear progress indicators and refresh
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            st.rerun()
            
        except Exception as e:
            self.logger.error(f"Error in custom answer analysis: {str(e)}")
            st.error(f"❌ Analysis failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    def _render_answer_quality_results(self, results_data: Dict[str, Any]) -> None:
        """Render answer quality analysis results."""
        analysis = results_data["analysis"]
        analysis_type = results_data["analysis_type"]
        
        st.header("📊 Answer Quality Analysis Results")
        
        # Query and metadata
        st.subheader("📝 Query Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Query:** {analysis['query']}")
            st.markdown(f"**Analysis Type:** {analysis_type.title()}")
        
        with col2:
            st.markdown(f"**Timestamp:** {analysis['timestamp'][:19]}")
            if analysis.get('expected_chunks'):
                st.markdown(f"**Expected Chunks:** {analysis['expected_chunks']}")
        
        # User Impact Assessment (Most Important)
        st.subheader("👤 User Impact Assessment")
        
        if "user_impact" in analysis:
            user_impact = analysis["user_impact"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                completion_status = "✅" if user_impact.get("can_complete_task", False) else "❌"
                st.metric(
                    "Can Complete Task",
                    completion_status,
                    f"{user_impact.get('completion_confidence', 0):.1%} confidence"
                )
            
            with col2:
                frustration_risk = user_impact.get("user_frustration_risk", "unknown")
                risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(frustration_risk, "⚪")
                st.metric(
                    "Frustration Risk",
                    f"{risk_color} {frustration_risk.title()}"
                )
            
            with col3:
                success_prob = user_impact.get("task_success_probability", 0)
                st.metric(
                    "Success Probability",
                    f"{success_prob:.1%}"
                )
            
            # Required follow-up actions
            if user_impact.get("required_followup"):
                st.markdown("**Required Follow-up Actions:**")
                for action in user_impact["required_followup"]:
                    st.markdown(f"• {action}")
        
        # Gap Analysis
        st.subheader("🔍 Content Gap Analysis")
        
        if "gap_analysis" in analysis:
            gap_analysis = analysis["gap_analysis"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                coverage = gap_analysis.get("coverage_percentage", 0)
                st.metric(
                    "Content Coverage",
                    f"{coverage:.1f}%",
                    delta=f"{coverage - 100:.1f}%" if coverage < 100 else "Complete"
                )
                
                if gap_analysis.get("missing_chunks"):
                    st.markdown("**Missing Chunks:**")
                    st.code(str(gap_analysis["missing_chunks"]))
            
            with col2:
                if gap_analysis.get("missing_content_areas"):
                    st.markdown("**Missing Content Areas:**")
                    for area in gap_analysis["missing_content_areas"]:
                        st.markdown(f"• {area}")
                
                if gap_analysis.get("critical_gaps"):
                    st.markdown("**Critical Gaps:**")
                    for gap in gap_analysis["critical_gaps"]:
                        st.error(gap)
        
        # Current Answer Analysis
        st.subheader("🤖 Current Answer Analysis")
        
        if "answer_analysis" in analysis and "answer" in analysis["answer_analysis"]:
            answer_analysis = analysis["answer_analysis"]
            answer = answer_analysis["answer"]
            
            # Show current answer
            st.markdown("**Current LLM Answer:**")
            st.text_area(
                "Current Answer",
                answer,
                height=200,
                disabled=True,
                label_visibility="collapsed"
            )
            
            # Answer quality metrics
            col1, col2, col3 = st.columns(3)
            
            if "completeness_analysis" in answer_analysis:
                completeness = answer_analysis["completeness_analysis"]
                with col1:
                    st.metric(
                        "Completeness Score",
                        f"{completeness.get('completeness_score', 0):.1%}"
                    )
            
            if "actionability_analysis" in answer_analysis:
                actionability = answer_analysis["actionability_analysis"]
                with col2:
                    st.metric(
                        "Actionability Score",
                        f"{actionability.get('actionability_score', 0):.1%}"
                    )
            
            if "user_value_assessment" in answer_analysis:
                user_value = answer_analysis["user_value_assessment"]
                with col3:
                    st.metric(
                        "User Value Score",
                        f"{user_value.get('user_value_score', 0):.1%}"
                    )
                
                if "value_assessment" in user_value:
                    st.markdown(f"**Assessment:** {user_value['value_assessment']}")
        
        # Retrieval Analysis
        with st.expander("🔍 Retrieval Analysis Details", expanded=False):
            if "retrieval_analysis" in analysis:
                retrieval = analysis["retrieval_analysis"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Context Length:** {retrieval.get('context_length', 0):,} chars")
                    st.markdown(f"**Chunk Count:** {retrieval.get('chunk_count', 0)}")
                    st.markdown(f"**Retrieved Chunks:** {retrieval.get('retrieved_chunk_ids', [])}")
                
                with col2:
                    if "total_chunks_available" in retrieval:
                        st.markdown(f"**Total Chunks Available:** {retrieval['total_chunks_available']}")
                
                # Show chunk analysis if available
                if "chunk_analysis" in retrieval and "score_distribution" in retrieval["chunk_analysis"]:
                    st.markdown("**Similarity Score Distribution:**")
                    dist = retrieval["chunk_analysis"]["score_distribution"]
                    score_data = {
                        "Min Score": f"{dist.get('min_score', 0):.3f}",
                        "Max Score": f"{dist.get('max_score', 0):.3f}",
                        "Mean Score": f"{dist.get('mean_score', 0):.3f}",
                        "Scores > 0.8": dist.get('scores_above_0_8', 0),
                        "Scores > 0.5": dist.get('scores_above_0_5', 0),
                        "Scores > 0.3": dist.get('scores_above_0_3', 0)
                    }
                    st.json(score_data)
        
        # Recommendations
        st.subheader("💡 Actionable Recommendations")
        
        if "recommendations" in analysis and analysis["recommendations"]:
            for i, recommendation in enumerate(analysis["recommendations"], 1):
                st.markdown(f"{i}. {recommendation}")
        else:
            st.info("No specific recommendations generated.")
        
        # Export options
        st.subheader("📥 Export Analysis")
        
        if st.button("📄 Download Analysis JSON"):
            st.download_button(
                label="Download Analysis",
                data=str(analysis),
                file_name=f"answer_quality_analysis_{analysis_type}_{int(time.time())}.json",
                mime="application/json"
            )
