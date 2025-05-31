"""Testing Controller for WorkApp2.

Handles UI integration for systematic engine testing and comparison.
"""

import streamlit as st
import logging
import time
from typing import Dict, Any, List, Optional

from utils.testing import TestRunner, TEST_CONFIGURATIONS
from utils.testing.engine_comparison import ComparisonResults

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
