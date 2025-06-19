import streamlit as st
import os
import sys
import json
import time
import asyncio
import subprocess
import threading
import queue
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import io
import contextlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the agent classes
try:
    from agent1_websearch import Agent1WebContentSearcher
    from agent2 import AIDatasetGenerator
    from agent3 import AIQualityAnalyzer
    from agent4 import GeminiAgent4DatasetEnhancer
except ImportError as e:
    st.error(f"Failed to import agent modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Testset Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .agent-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-completed {
        color: #007bff;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-pending {
        color: #6c757d;
        font-weight: bold;
    }
    
    .progress-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .log-container {
        background: #1e1e1e;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        max-height: 400px;
        overflow-y: auto;
        margin: 1rem 0;
        white-space: pre-wrap;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitLogger:
    """Custom logger that captures output for Streamlit display"""
    
    def __init__(self):
        self.logs = []
        self.max_logs = 1000
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.logs.append(log_entry)
        
        # Keep only the last max_logs entries
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
    
    def get_logs(self) -> List[str]:
        return self.logs
    
    def clear(self):
        self.logs = []

class OutputCapture:
    """Capture stdout and stderr for real-time display"""
    
    def __init__(self):
        self.captured_output = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def start_capture(self):
        """Start capturing output"""
        self.string_io = io.StringIO()
        sys.stdout = self.string_io
        sys.stderr = self.string_io
    
    def stop_capture(self):
        """Stop capturing output and return captured text"""
        captured = self.string_io.getvalue()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        return captured
    
    def get_output(self):
        """Get current captured output without stopping capture"""
        return self.string_io.getvalue()

class AgentPipeline:
    """Main pipeline class that manages the execution of all agents"""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.logger = StreamlitLogger()
        self.output_capture = OutputCapture()
        self.agents_status = {
            "Agent 1": {"status": "pending", "progress": 0, "output": None, "description": "Dataset Web Search"},
            "Agent 2": {"status": "pending", "progress": 0, "output": None, "description": "Dataset Generation"},
            "Agent 3": {"status": "pending", "progress": 0, "output": None, "description": "Quality Analysis"},
            "Agent 4": {"status": "pending", "progress": 0, "output": None, "description": "Enhancement & Resolution"}
        }
        self.start_time = None
        self.end_time = None
        self.current_agent = None
        
        # Create output directories
        os.makedirs("agent2_output", exist_ok=True)
        os.makedirs("agent3_output", exist_ok=True)
        os.makedirs("agent4_output", exist_ok=True)
        
    def update_agent_status(self, agent_name: str, status: str, progress: int = 0, output: Any = None):
        """Update the status of a specific agent"""
        self.agents_status[agent_name]["status"] = status
        self.agents_status[agent_name]["progress"] = progress
        if output is not None:
            self.agents_status[agent_name]["output"] = output
        self.current_agent = agent_name if status == "running" else None
    
    async def run_pipeline(self, query: str, progress_callback=None, log_callback=None):
        """Run the complete pipeline"""
        self.start_time = datetime.now()
        self.logger.log(f"Starting pipeline for query: {query}")
        
        try:
            # Agent 1: Web Search
            self.update_agent_status("Agent 1", "running", 10)
            if log_callback:
                log_callback("🔍 Starting Agent 1: Dataset Web Search...")
            
            agent1_output = await self.run_agent1(query, log_callback)
            
            self.update_agent_status("Agent 1", "completed", 100, agent1_output)
            if log_callback:
                log_callback("✅ Agent 1 completed successfully")
            
            # Agent 2: Dataset Generation
            self.update_agent_status("Agent 2", "running", 10)
            if log_callback:
                log_callback("🏗️ Starting Agent 2: Dataset Generation...")
            
            agent2_output = await self.run_agent2("agent1_output.json", log_callback)
            
            self.update_agent_status("Agent 2", "completed", 100, agent2_output)
            if log_callback:
                log_callback("✅ Agent 2 completed successfully")
            
            # Agent 3: Quality Analysis
            self.update_agent_status("Agent 3", "running", 10)
            if log_callback:
                log_callback("🔍 Starting Agent 3: Quality Analysis...")
            
            agent3_output = await self.run_agent3("agent2_output/agent2_ai_output.json", log_callback)
            
            self.update_agent_status("Agent 3", "completed", 100, agent3_output)
            if log_callback:
                log_callback("✅ Agent 3 completed successfully")
            
            # Agent 4: Enhancement
            self.update_agent_status("Agent 4", "running", 10)
            if log_callback:
                log_callback("🚀 Starting Agent 4: Dataset Enhancement...")
            
            agent4_output = await self.run_agent4(log_callback)
            
            self.update_agent_status("Agent 4", "completed", 100, agent4_output)
            if log_callback:
                log_callback("✅ Agent 4 completed successfully")
            
            self.end_time = datetime.now()
            total_time = (self.end_time - self.start_time).total_seconds()
            
            if log_callback:
                log_callback(f"🎉 Pipeline completed in {total_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.log(error_msg, "ERROR")
            if log_callback:
                log_callback(f"❌ {error_msg}")
            
            # Mark current agent as error
            if self.current_agent:
                self.update_agent_status(self.current_agent, "error")
            
            return False
    
    async def run_agent1(self, query: str, log_callback=None):
        """Run Agent 1 with progress tracking"""
        try:
            if log_callback:
                log_callback("Agent 1: Initializing web search agent...")
            
            agent1 = Agent1WebContentSearcher(self.gemini_api_key)
            
            self.update_agent_status("Agent 1", "running", 25)
            if log_callback:
                log_callback("Agent 1: Analyzing user context...")
            
            context = await agent1.analyze_user_context(query)
            
            self.update_agent_status("Agent 1", "running", 50)
            if log_callback:
                log_callback("Agent 1: Searching for relevant datasets...")
            
            datasets = await agent1.search_for_datasets(context, max_datasets=3)
            
            self.update_agent_status("Agent 1", "running", 75)
            if log_callback:
                log_callback("Agent 1: Generating recommendations...")
            
            recommendations = await agent1.generate_recommendations(context, datasets)
            
            # Save output
            output = {
                "user_context": context.__dict__,
                "available_datasets": [dataset.__dict__ for dataset in datasets],
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
            with open("agent1_output.json", "w") as f:
                json.dump(output, f, indent=2, default=str)
            
            if log_callback:
                log_callback(f"Agent 1: Found {len(datasets)} datasets and saved output")
            
            return output
            
        except Exception as e:
            if log_callback:
                log_callback(f"Agent 1 Error: {str(e)}")
            raise
    
    async def run_agent2(self, agent1_output_path: str, log_callback=None):
        """Run Agent 2 with progress tracking"""
        try:
            if log_callback:
                log_callback("Agent 2: Initializing dataset generator...")
            
            agent2 = AIDatasetGenerator(self.gemini_api_key)
            
            self.update_agent_status("Agent 2", "running", 30)
            if log_callback:
                log_callback("Agent 2: Processing Agent 1 output...")
            
            output = await agent2.process_agent1_output(agent1_output_path)
            
            self.update_agent_status("Agent 2", "running", 75)
            if log_callback:
                log_callback(f"Agent 2: Generated {len(output.generated_datasets)} datasets")
            
            return output
            
        except Exception as e:
            if log_callback:
                log_callback(f"Agent 2 Error: {str(e)}")
            raise
    
    async def run_agent3(self, agent2_output_path: str, log_callback=None):
        """Run Agent 3 with progress tracking"""
        try:
            if log_callback:
                log_callback("Agent 3: Initializing quality analyzer...")
            
            agent3 = AIQualityAnalyzer(self.gemini_api_key)
            
            self.update_agent_status("Agent 3", "running", 40)
            if log_callback:
                log_callback("Agent 3: Analyzing dataset quality...")
            
            output = await agent3.analyze_agent2_output(agent2_output_path)
            
            self.update_agent_status("Agent 3", "running", 80)
            
            # Save output
            with open("agent3_output/agent3_analysis.json", "w") as f:
                json.dump(output.to_dict(), f, indent=2, default=str)
            
            if log_callback:
                log_callback(f"Agent 3: Found {len(output.critical_issues)} critical issues")
            
            return output
            
        except Exception as e:
            if log_callback:
                log_callback(f"Agent 3 Error: {str(e)}")
            raise
    
    async def run_agent4(self, log_callback=None):
        """Run Agent 4 with progress tracking"""
        try:
            if log_callback:
                log_callback("Agent 4: Initializing enhancement agent...")
            
            agent4 = GeminiAgent4DatasetEnhancer(self.gemini_api_key)
            
            self.update_agent_status("Agent 4", "running", 25)
            if log_callback:
                log_callback("Agent 4: Loading datasets and analysis...")
            
            datasets = await agent4.load_agent2_datasets()
            
            self.update_agent_status("Agent 4", "running", 50)
            if log_callback:
                log_callback("Agent 4: Loading quality analysis...")
            
            agent3_issues = agent4.load_agent3_reports()
            
            self.update_agent_status("Agent 4", "running", 75)
            if log_callback:
                log_callback("Agent 4: Enhancing datasets with Gemini AI...")
            
            enhanced_output = await agent4.run_gemini_enhancement_pipeline()
            
            if log_callback:
                log_callback("Agent 4: Enhancement completed")
            
            return enhanced_output
            
        except Exception as e:
            if log_callback:
                log_callback(f"Agent 4 Error: {str(e)}")
            raise

def display_agent_status_sidebar(pipeline):
    """Display agent status in sidebar"""
    st.sidebar.header("📊 Agent Status")
    
    for agent_name, status_info in pipeline.agents_status.items():
        status = status_info["status"]
        progress = status_info["progress"]
        description = status_info["description"]
        
        with st.sidebar.container():
            if status == "pending":
                st.markdown(f"⏳ **{agent_name}**")
                st.caption(description)
                st.markdown("<span class='status-pending'>Pending</span>", unsafe_allow_html=True)
            elif status == "running":
                st.markdown(f"🔄 **{agent_name}**")
                st.caption(description)
                st.markdown("<span class='status-running'>Running</span>", unsafe_allow_html=True)
                st.progress(progress / 100)
            elif status == "completed":
                st.markdown(f"✅ **{agent_name}**")
                st.caption(description)
                st.markdown("<span class='status-completed'>Completed</span>", unsafe_allow_html=True)
            elif status == "error":
                st.markdown(f"❌ **{agent_name}**")
                st.caption(description)
                st.markdown("<span class='status-error'>Error</span>", unsafe_allow_html=True)
            
            st.markdown("---")

def display_results():
    """Display pipeline results"""
    st.markdown("### 📊 Pipeline Results")
    
    # Display generated datasets
    if os.path.exists("agent2_output"):
        st.markdown("#### 📁 Generated Datasets")
        
        csv_files = [f for f in os.listdir("agent2_output") if f.endswith(".csv")]
        
        if csv_files:
            for file in csv_files:
                file_path = os.path.join("agent2_output", file)
                try:
                    df = pd.read_csv(file_path)
                    
                    with st.expander(f"📄 {file} ({len(df)} rows × {len(df.columns)} columns)"):
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # File info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", len(df))
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            file_size = os.path.getsize(file_path) / 1024  # KB
                            st.metric("Size", f"{file_size:.1f} KB")
                        
                        # Download button
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label=f"⬇️ Download {file}",
                                data=f.read(),
                                file_name=file,
                                mime="text/csv",
                                key=f"download_{file}"
                            )
                except Exception as e:
                    st.error(f"Error reading {file}: {str(e)}")
        else:
            st.info("No CSV files found in agent2_output directory")
    
    # Display quality analysis
    if os.path.exists("agent3_output/agent3_analysis.json"):
        st.markdown("#### 🔍 Quality Analysis Results")
        
        try:
            with open("agent3_output/agent3_analysis.json", "r") as f:
                analysis = json.load(f)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            processing_stats = analysis.get("processing_stats", {})
            
            with col1:
                st.metric("Datasets Analyzed", processing_stats.get("datasets_analyzed", 0))
            with col2:
                st.metric("Total Issues", processing_stats.get("total_issues_found", 0))
            with col3:
                st.metric("Critical Issues", processing_stats.get("critical_issues_count", 0))
            with col4:
                processing_time = processing_stats.get("processing_time_seconds", 0)
                st.metric("Analysis Time", f"{processing_time:.2f}s")
            
            # Critical issues
            critical_issues = analysis.get("critical_issues", [])
            if critical_issues:
                st.markdown("##### ⚠️ Critical Issues Found")
                for i, issue in enumerate(critical_issues[:5]):  # Show first 5
                    with st.expander(f"Issue {i+1}: {issue.get('category', 'Unknown')}"):
                        st.write(f"**Severity:** {issue.get('severity', 'Unknown')}")
                        st.write(f"**Description:** {issue.get('description', 'No description')}")
                        st.write(f"**Column:** {issue.get('column', 'N/A')}")
                        st.write(f"**Recommendation:** {issue.get('ai_recommendation', 'No recommendation')}")
            
            # Overall assessment
            overall_assessment = analysis.get("overall_assessment", {})
            if overall_assessment:
                st.markdown("##### 📈 Overall Assessment")
                st.json(overall_assessment)
                
        except Exception as e:
            st.error(f"Error reading analysis results: {str(e)}")
    
    # Display enhancement results
    if os.path.exists("agent4_output"):
        st.markdown("#### 🚀 Enhancement Results")
        
        enhanced_files = [f for f in os.listdir("agent4_output") if f.endswith(".csv")]
        
        if enhanced_files:
            st.success(f"✅ {len(enhanced_files)} datasets enhanced and saved to agent4_output/")
            
            for file in enhanced_files:
                file_path = os.path.join("agent4_output", file)
                try:
                    df = pd.read_csv(file_path)
                    
                    with st.expander(f"🔧 Enhanced: {file}"):
                        st.dataframe(df.head(5), use_container_width=True)
                        
                        # Download button
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label=f"⬇️ Download Enhanced {file}",
                                data=f.read(),
                                file_name=f"enhanced_{file}",
                                mime="text/csv",
                                key=f"download_enhanced_{file}"
                            )
                except Exception as e:
                    st.error(f"Error reading enhanced file {file}: {str(e)}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Testset Pipeline</h1>
        <p>Intelligent 4-Agent System for Testset Generation, Analysis & Enhancement</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if not gemini_api_key:
            st.warning("Please enter your Gemini API key to proceed")
            st.stop()
        
        # Initialize session state
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = AgentPipeline(gemini_api_key)
        
        if 'pipeline_running' not in st.session_state:
            st.session_state.pipeline_running = False
        
        # Display agent status
        display_agent_status_sidebar(st.session_state.pipeline)
        
        # Clear button
        if st.button("🔄 Reset Pipeline"):
            st.session_state.pipeline = AgentPipeline(gemini_api_key)
            st.session_state.pipeline_running = False
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Testset Requirements")
        
        # User input
        query = st.text_area(
            "Enter your Test requirements:",
            placeholder="I want Testset on effect of gallbladder stones on human body",
            height=100,
            disabled=st.session_state.pipeline_running
        )
        
        # Submit button
        start_button = st.button(
            "🚀 Start Pipeline", 
            type="primary", 
            disabled=not query.strip() or st.session_state.pipeline_running
        )
        
        if start_button and query.strip():
            st.session_state.pipeline_running = True
            
            # Create containers for real-time updates
            progress_container = st.container()
            log_container = st.container()
            
            with progress_container:
                st.markdown("### 📈 Pipeline Progress")
                overall_progress = st.progress(0)
                current_status = st.empty()
            
            with log_container:
                st.markdown("### 📝 Live Logs")
                log_display = st.empty()
            
            # Define callback functions
            def update_logs(message):
                st.session_state.pipeline.logger.log(message)
                logs = st.session_state.pipeline.logger.get_logs()
                log_text = "\\n".join(logs[-15:])  # Show last 15 logs
                log_display.markdown(f"""
                <div class="log-container">
                    {log_text}
                </div>
                """, unsafe_allow_html=True)
            
            # Run pipeline
            async def run_pipeline_async():
                return await st.session_state.pipeline.run_pipeline(
                    query, 
                    log_callback=update_logs
                )
            
            # Execute pipeline
            try:
                # Update progress based on agent status
                def update_progress():
                    completed_agents = sum(1 for agent in st.session_state.pipeline.agents_status.values() 
                                         if agent["status"] == "completed")
                    total_agents = len(st.session_state.pipeline.agents_status)
                    progress = (completed_agents / total_agents) * 100
                    overall_progress.progress(int(progress))
                    
                    # Update current status
                    current_agent = st.session_state.pipeline.current_agent
                    if current_agent:
                        agent_info = st.session_state.pipeline.agents_status[current_agent]
                        current_status.text(f"Running: {current_agent} - {agent_info['description']}")
                    elif completed_agents == total_agents:
                        current_status.text("✅ All agents completed successfully!")
                    else:
                        current_status.text("⏳ Preparing to start...")
                
                # Start progress monitoring
                progress_placeholder = st.empty()
                
                # Run the pipeline
                success = asyncio.run(run_pipeline_async())
                
                # Final progress update
                update_progress()
                
                st.session_state.pipeline_running = False
                
                if success:
                    overall_progress.progress(100)
                    st.markdown("""
                    <div class="success-message">
                        🎉 <strong>Pipeline completed successfully!</strong> 
                        Check the results below and download your generated datasets.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display results
                    display_results()
                    
                else:
                    st.markdown("""
                    <div class="error-message">
                        ❌ <strong>Pipeline failed.</strong> 
                        Check the logs above for error details.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Update progress during execution
                while st.session_state.pipeline_running:
                    update_progress()
                    time.sleep(1)
                
            except Exception as e:
                st.session_state.pipeline_running = False
                st.markdown(f"""
                <div class="error-message">
                    ❌ <strong>Error running pipeline:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.header("⏱️ Execution Timer")
        
        # Timer display
        if st.session_state.pipeline.start_time:
            if st.session_state.pipeline.end_time:
                total_time = (st.session_state.pipeline.end_time - st.session_state.pipeline.start_time).total_seconds()
                st.metric("Total Time", f"{total_time:.2f}s")
            elif st.session_state.pipeline_running:
                # Real-time timer
                timer_placeholder = st.empty()
                if st.session_state.pipeline.start_time:
                    current_time = datetime.now()
                    elapsed_time = (current_time - st.session_state.pipeline.start_time).total_seconds()
                    timer_placeholder.metric("Elapsed Time", f"{elapsed_time:.1f}s")
        
        st.header("📈 Pipeline Overview")
        
        # Agent progress overview
        for agent_name, status_info in st.session_state.pipeline.agents_status.items():
            status = status_info["status"]
            progress = status_info["progress"]
            description = status_info["description"]
            
            with st.container():
                st.markdown(f"**{agent_name}**")
                st.caption(description)
                
                if status == "running":
                    st.progress(progress / 100)
                elif status == "completed":
                    st.progress(100)
                else:
                    st.progress(0)
                
                # Status badge
                if status == "pending":
                    st.markdown("🔘 Pending")
                elif status == "running":
                    st.markdown("🔄 Running")
                elif status == "completed":
                    st.markdown("✅ Completed")
                elif status == "error":
                    st.markdown("❌ Error")
                
                st.markdown("---")
        
        # System info
        st.header("ℹ️ System Info")
        st.info(f"""
        **Python Version:** {sys.version.split()[0]}
        **Working Directory:** {os.getcwd()}
        **Available Agents:** 4
        """)

if __name__ == "__main__":
    main()

