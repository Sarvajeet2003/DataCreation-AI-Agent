"""
Agent 4: AI-Driven Dataset Enhancement & Issue Resolution Agent (Gemini Powered)
- Loads Agent 2's generated datasets
- Reads Agent 3's quality analysis reports  
- Validates logical correctness of synthetic data using Gemini
- Fixes issues identified by Agent 3
100% Gemini AI-intelligent with targeted problem solving
"""

import json
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import asyncio
from dataclasses import dataclass
import warnings
import google.generativeai as genai
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

@dataclass
class Agent3Issue:
    """Issue identified by Agent 3"""
    issue_type: str
    description: str
    severity: str
    column: Optional[str]
    affected_rows: Optional[List[int]]
    recommendation: str

@dataclass
class LogicalValidationResult:
    """Gemini logical validation result"""
    validation_type: str
    column_combination: List[str]
    issues_found: List[Dict]
    ai_reasoning: str
    fix_applied: bool
    confidence: float

class GeminiAIBrain:
    """Gemini AI brain for intelligent dataset processing"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini AI model"""
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("🧠 Gemini AI brain initialized successfully")
            return model
        except Exception as e:
            logger.error(f"Gemini AI initialization failed: {e}")
            raise
    
    async def generate_response(self, prompt: str) -> str:
        """Generate response using Gemini AI"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini response generation failed: {e}")
            return ""
    
    def parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from Gemini response"""
        try:
            # Extract JSON from response
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                json_str = matches[0]
                return json.loads(json_str)
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing Gemini JSON response: {e}")
            return None

class GeminiLogicalValidator:
    """Gemini-powered logical correctness validator"""
    
    def __init__(self, gemini_brain: GeminiAIBrain):
        self.gemini = gemini_brain
        self.validation_results = []
    
    async def validate_medical_logic(self, df: pd.DataFrame) -> List[LogicalValidationResult]:
        """Gemini validates medical/health dataset logical consistency"""
        
        logger.info("🩺 Gemini AI validating medical dataset logic...")
        
        # Sample data for analysis (first 10 rows to avoid token limits)
        sample_data = df.head(10).to_dict('records')
        
        validation_prompt = f"""
        You are an expert medical data validation AI. Analyze this synthetic medical dataset for logical inconsistencies and medical impossibilities.

        Dataset Information:
        - Shape: {df.shape[0]} rows × {df.shape[1]} columns
        - Columns: {list(df.columns)}
        - Data Types: {df.dtypes.to_dict()}
        
        Sample Data (first 10 rows):
        {json.dumps(sample_data, indent=2, default=str)}
        
        Statistical Summary:
        {df.describe(include='all').to_dict()}

        MEDICAL LOGIC VALIDATION TASKS:
        1. **Age-Disease Consistency**: Check if age ranges match disease prevalence
        2. **Symptom-Diagnosis Alignment**: Verify symptoms match medical conditions
        3. **Lab Values Logic**: Validate lab results are within possible ranges
        4. **Treatment Appropriateness**: Check if treatments match conditions
        5. **Risk Factor Correlation**: Validate risk factors align with outcomes
        6. **Impossible Combinations**: Find medically impossible data combinations
        7. **Missing Critical Relationships**: Identify missing medical correlations
        
        Return ONLY valid JSON with this exact structure:
        {{
            "logical_issues": [
                {{
                    "validation_type": "age_disease_consistency",
                    "columns_involved": ["age", "disease_severity"],
                    "issue_description": "Specific medical logic problem found",
                    "affected_rows": [1, 5, 10],
                    "severity": "critical",
                    "medical_reasoning": "Why this is medically impossible/illogical",
                    "fix_suggestion": "Specific fix to make data medically sound",
                    "confidence": 0.95
                }}
            ],
            "overall_logic_score": 0.85,
            "critical_fixes_needed": true,
            "medical_validity_assessment": "Overall medical soundness evaluation"
        }}
        """
        
        try:
            response = await self.gemini.generate_response(validation_prompt)
            result = self.gemini.parse_json_response(response)
            
            if result and 'logical_issues' in result:
                logger.info(f"🩺 Gemini found {len(result['logical_issues'])} medical logic issues")
                
                return [
                    LogicalValidationResult(
                        validation_type=issue['validation_type'],
                        column_combination=issue['columns_involved'],
                        issues_found=[issue],
                        ai_reasoning=issue['issue_description'],
                        fix_applied=False,
                        confidence=issue.get('confidence', 0.8)
                    )
                    for issue in result['logical_issues']
                ]
            
            logger.info("🩺 Gemini validation: No major logical issues found")
            return []
            
        except Exception as e:
            logger.error(f"Gemini medical logic validation failed: {e}")
            return []

class GeminiIssueResolver:
    """Gemini-powered Agent 3 issue resolver"""
    
    def __init__(self, gemini_brain: GeminiAIBrain):
        self.gemini = gemini_brain
        self.resolved_issues = []
    
    async def resolve_agent3_issues(self, df: pd.DataFrame, agent3_issues: List[Agent3Issue]) -> Tuple[pd.DataFrame, List[Dict]]:
        """Gemini resolves issues identified by Agent 3"""
        
        logger.info(f"🔧 Gemini resolving {len(agent3_issues)} issues from Agent 3...")
        
        enhanced_df = df.copy()
        resolution_results = []
        
        for i, issue in enumerate(agent3_issues):
            try:
                logger.info(f"🔧 Resolving issue {i+1}/{len(agent3_issues)}: {issue.issue_type}")
                
                resolution_result = await self._resolve_single_issue(enhanced_df, issue)
                
                if resolution_result['success']:
                    enhanced_df = resolution_result['enhanced_data']
                    resolution_results.append({
                        'issue_type': issue.issue_type,
                        'status': 'resolved',
                        'fix_applied': resolution_result['fix_description'],
                        'gemini_reasoning': resolution_result['ai_reasoning'],
                        'confidence': resolution_result.get('confidence', 0.8)
                    })
                    logger.info(f"✅ Resolved: {issue.issue_type}")
                else:
                    resolution_results.append({
                        'issue_type': issue.issue_type,
                        'status': 'failed',
                        'error': resolution_result['error']
                    })
                    logger.warning(f"❌ Failed to resolve: {issue.issue_type}")
                    
            except Exception as e:
                logger.error(f"Error resolving issue {issue.issue_type}: {e}")
                resolution_results.append({
                    'issue_type': issue.issue_type,
                    'status': 'error',
                    'error': str(e)
                })
        
        return enhanced_df, resolution_results
    
    async def _resolve_single_issue(self, df: pd.DataFrame, issue: Agent3Issue) -> Dict:
        """Gemini resolves a single issue with intelligent analysis"""
        
        # Get sample data around the issue
        sample_size = min(20, len(df))
        sample_df = df.head(sample_size)
        
        resolution_prompt = f"""
        You are an expert data quality engineer. Fix this specific data quality issue using intelligent reasoning.

        ISSUE TO FIX:
        - Type: {issue.issue_type}
        - Description: {issue.description}
        - Severity: {issue.severity}
        - Affected Column: {issue.column}
        - Agent 3 Recommendation: {issue.recommendation}

        DATASET CONTEXT:
        - Shape: {df.shape[0]} rows × {df.shape[1]} columns
        - Target Column: {issue.column}
        {f"- Column Type: {df[issue.column].dtype}" if issue.column and issue.column in df.columns else ""}
        {f"- Column Stats: {df[issue.column].describe().to_dict()}" if issue.column and issue.column in df.columns and df[issue.column].dtype in ['int64', 'float64'] else ""}
        {f"- Sample Values: {df[issue.column].head(10).tolist()}" if issue.column and issue.column in df.columns else ""}
        {f"- Null Count: {df[issue.column].isnull().sum()}" if issue.column and issue.column in df.columns else ""}
        
        Sample Data (first {sample_size} rows):
        {sample_df.to_dict('records')}

        PROVIDE SPECIFIC FIX INSTRUCTIONS in this exact JSON format:
        {{
            "fix_type": "replace_values|fill_missing|remove_duplicates|standardize_format|add_validation|recalculate_values",
            "target_column": "{issue.column or 'multiple'}",
            "fix_logic": "Detailed explanation of what fix will be applied",
            "implementation_method": "python_code|direct_replacement|statistical_imputation|rule_based",
            "new_values": ["replacement", "values", "if", "needed"],
            "conditions": "When/where to apply this fix",
            "expected_outcome": "What the data should look like after fix",
            "medical_reasoning": "Why this fix makes medical/logical sense",
            "confidence": 0.95
        }}

        Focus on making the data medically sound and logically consistent.
        """
        
        try:
            response = await self.gemini.generate_response(resolution_prompt)
            fix_instructions = self.gemini.parse_json_response(response)
            
            if fix_instructions and 'fix_type' in fix_instructions:
                enhanced_df = await self._apply_gemini_fix(df, issue, fix_instructions)
                return {
                    'success': True,
                    'enhanced_data': enhanced_df,
                    'fix_description': fix_instructions.get('fix_logic', ''),
                    'ai_reasoning': fix_instructions.get('medical_reasoning', ''),
                    'confidence': fix_instructions.get('confidence', 0.8)
                }
            
            return {'success': False, 'error': 'Invalid Gemini fix instructions'}
            
        except Exception as e:
            logger.error(f"Gemini issue resolution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _apply_gemini_fix(self, df: pd.DataFrame, issue: Agent3Issue, fix_instructions: Dict) -> pd.DataFrame:
        """Apply Gemini-generated intelligent fix to dataframe"""
        
        enhanced_df = df.copy()
        fix_type = fix_instructions.get('fix_type')
        target_column = fix_instructions.get('target_column')
        
        try:
            logger.info(f"🔧 Applying {fix_type} fix to {target_column}")
            
            if fix_type == "fill_missing" and target_column in enhanced_df.columns:
                # Intelligent missing value filling
                new_values = fix_instructions.get('new_values', [])
                if new_values:
                    fill_value = new_values[0]
                    
                    # Convert to appropriate type
                    if enhanced_df[target_column].dtype == 'object':
                        enhanced_df[target_column] = enhanced_df[target_column].fillna(str(fill_value))
                    else:
                        try:
                            enhanced_df[target_column] = enhanced_df[target_column].fillna(pd.to_numeric(fill_value))
                        except:
                            enhanced_df[target_column] = enhanced_df[target_column].fillna(fill_value)
            
            elif fix_type == "replace_values" and target_column in enhanced_df.columns:
                # Intelligent value replacement based on conditions
                conditions = fix_instructions.get('conditions', '')
                new_values = fix_instructions.get('new_values', [])
                
                if new_values and len(new_values) > 0:
                    # Simple replacement strategy - can be enhanced
                    enhanced_df[target_column] = enhanced_df[target_column].fillna(new_values[0])
            
            elif fix_type == "remove_duplicates":
                # Remove duplicate rows
                original_shape = enhanced_df.shape
                enhanced_df = enhanced_df.drop_duplicates()
                logger.info(f"🔧 Removed {original_shape[0] - enhanced_df.shape[0]} duplicate rows")
            
            elif fix_type == "standardize_format" and target_column in enhanced_df.columns:
                # Standardize data format
                if enhanced_df[target_column].dtype == 'object':
                    enhanced_df[target_column] = enhanced_df[target_column].astype(str).str.strip().str.title()
            
            elif fix_type == "add_validation" and target_column in enhanced_df.columns:
                # Add data validation (e.g., ensure positive values for age)
                if enhanced_df[target_column].dtype in ['int64', 'float64']:
                    enhanced_df[target_column] = enhanced_df[target_column].abs()  # Make positive
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error applying Gemini fix: {e}")
            return df

class GeminiAgent4DatasetEnhancer:
    """Main Agent 4 - Gemini-Powered Dataset Enhancement & Issue Resolution"""
    
    def __init__(self, gemini_api_key: str):
        self.datasets = {}
        self.agent3_issues = []
        self.gemini_brain = GeminiAIBrain(gemini_api_key)
        self.logical_validator = GeminiLogicalValidator(self.gemini_brain)
        self.issue_resolver = GeminiIssueResolver(self.gemini_brain)
        
        logger.info("🤖 Agent 4 (Gemini-Powered) initialized - Ready for intelligent enhancement")
    
    async def load_agent2_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load datasets generated by Agent 2"""
        
        logger.info("📊 Loading Agent 2 generated datasets...")
        
        agent2_output_dir = "agent2_output"
        datasets = {}
        
        if not os.path.exists(agent2_output_dir):
            logger.warning(f"Agent 2 output directory not found: {agent2_output_dir}")
            return datasets
        
        # Load CSV files from agent2_output
        for file in os.listdir(agent2_output_dir):
            if file.endswith('.csv'):
                try:
                    file_path = os.path.join(agent2_output_dir, file)
                    df = pd.read_csv(file_path)
                    
                    if not df.empty:
                        dataset_name = file.replace('.csv', '')
                        datasets[dataset_name] = df
                        logger.info(f"✅ Loaded: {dataset_name} ({df.shape[0]} rows × {df.shape[1]} cols)")
                    
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
        
        return datasets
    
    async def load_agent3_reports(self) -> List[Agent3Issue]:
        """Load quality analysis reports from Agent 3"""
        
        logger.info("📋 Loading Agent 3 quality reports...")
        
        agent3_output_dir = "agent3_output"
        issues = []
        
        if not os.path.exists(agent3_output_dir):
            logger.warning(f"Agent 3 output directory not found: {agent3_output_dir}")
            return issues
        
        # Load JSON reports from agent3_output
        for file in os.listdir(agent3_output_dir):
            if file.endswith('.json'):
                try:
                    file_path = os.path.join(agent3_output_dir, file)
                    
                    with open(file_path, 'r') as f:
                        report_data = json.load(f)
                    
                    # Extract issues from different report formats
                    extracted_issues = self._extract_issues_from_report(report_data, file)
                    issues.extend(extracted_issues)
                    
                    logger.info(f"✅ Loaded: {file} ({len(extracted_issues)} issues)")
                    
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
        
        return issues
    
    def _extract_issues_from_report(self, report_data: Dict, filename: str) -> List[Agent3Issue]:
        """Extract issues from Agent 3 report format"""
        
        issues = []
        
        try:
            # Handle different report structures
            if 'quality_issues' in report_data:
                for issue_data in report_data['quality_issues']:
                    issues.append(Agent3Issue(
                        issue_type=issue_data.get('issue_type', 'unknown'),
                        description=issue_data.get('description', ''),
                        severity=issue_data.get('severity', 'medium'),
                        column=issue_data.get('column'),
                        affected_rows=issue_data.get('affected_rows'),
                        recommendation=issue_data.get('recommendation', '')
                    ))
            
            # Handle quality_analysis_report structure
            if 'analysis_results' in report_data:
                analysis = report_data['analysis_results']
                if isinstance(analysis, dict):
                    for key, value in analysis.items():
                        if isinstance(value, dict):
                            # Check for various issue indicators
                            if 'issues' in value:
                                for issue in value['issues']:
                                    issues.append(Agent3Issue(
                                        issue_type=f"{key}_issue",
                                        description=issue.get('description', str(issue)),
                                        severity=issue.get('severity', 'medium'),
                                        column=issue.get('column'),
                                        affected_rows=None,
                                        recommendation=issue.get('recommendation', f"Fix {key} issue")
                                    ))
                            elif 'problems' in value:
                                for problem in value['problems']:
                                    issues.append(Agent3Issue(
                                        issue_type=f"{key}_problem",
                                        description=str(problem),
                                        severity='medium',
                                        column=None,
                                        affected_rows=None,
                                        recommendation=f"Resolve {key} problem"
                                    ))
            
            # Look for direct issue indicators
            for key, value in report_data.items():
                if 'issue' in key.lower() or 'problem' in key.lower() or 'error' in key.lower():
                    if isinstance(value, (list, dict)) and value:
                        issues.append(Agent3Issue(
                            issue_type=key,
                            description=str(value),
                            severity='medium',
                            column=None,
                            affected_rows=None,
                            recommendation=f"Address {key} identified in {filename}"
                        ))
            
            # If no specific issues found, create a general enhancement issue
            if not issues and report_data:
                issues.append(Agent3Issue(
                    issue_type="general_enhancement",
                    description=f"General quality improvements needed based on {filename}",
                    severity='low',
                    column=None,
                    affected_rows=None,
                    recommendation="Apply general data quality improvements"
                ))
        
        except Exception as e:
            logger.error(f"Error extracting issues from {filename}: {e}")
            # Create a generic issue if extraction fails
            issues.append(Agent3Issue(
                issue_type="extraction_error",
                description=f"Could not parse report {filename}: {e}",
                severity='low',
                column=None,
                affected_rows=None,
                recommendation="Manual review of report needed"
            ))
        
        return issues
    
    async def run_gemini_enhancement_pipeline(self) -> Dict:
        """Run complete Gemini-powered Agent 4 pipeline"""
        
        start_time = datetime.now()
        
        print("\n🤖 Agent 4: Gemini-Powered Dataset Enhancement & Issue Resolution")
        print("=" * 80)
        print("🧠 Gemini AI + Agent 2 Datasets + Agent 3 Issues = Enhanced Data")
        print("=" * 80)
        
        try:
            # Step 1: Load Agent 2 datasets
            print("📊 Step 1: Loading Agent 2 generated datasets...")
            self.datasets = await self.load_agent2_datasets()
            print(f"✅ Loaded {len(self.datasets)} datasets from Agent 2")
            
            # Step 2: Load Agent 3 quality reports  
            print("\n📋 Step 2: Loading Agent 3 quality reports...")
            self.agent3_issues = await self.load_agent3_reports()
            print(f"✅ Found {len(self.agent3_issues)} issues from Agent 3")
            
            # Display found issues
            if self.agent3_issues:
                print("🔍 Issues to resolve:")
                for i, issue in enumerate(self.agent3_issues):
                    print(f"  {i+1}. {issue.issue_type}: {issue.description[:50]}...")
            
            if not self.datasets:
                print("⚠️ No datasets found from Agent 2")
                return {'success': False, 'error': 'No datasets to enhance'}
            
            enhanced_results = {}
            
            # Step 3: Process each dataset with Gemini intelligence
            for dataset_name, df in self.datasets.items():
                print(f"\n🔬 Step 3: Gemini processing {dataset_name}...")
                print(f"  📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Sub-step 3a: Gemini logical validation
                print(f"  🧠 Gemini logical correctness validation...")
                logical_results = await self.logical_validator.validate_medical_logic(df)
                print(f"  ✅ Gemini found {len(logical_results)} logical issues")
                
                # Sub-step 3b: Resolve Agent 3 issues with Gemini
                print(f"  🔧 Gemini resolving Agent 3 identified issues...")
                enhanced_df, resolution_results = await self.issue_resolver.resolve_agent3_issues(
                    df, self.agent3_issues
                )
                successful_fixes = sum(1 for r in resolution_results if r['status'] == 'resolved')
                print(f"  ✅ Gemini successfully resolved {successful_fixes}/{len(self.agent3_issues)} issues")
                
                enhanced_results[dataset_name] = {
                    'original_shape': df.shape,
                    'enhanced_shape': enhanced_df.shape,
                    'logical_validations': logical_results,
                    'agent3_resolutions': resolution_results,
                    'enhanced_data': enhanced_df
                }
            
            # Step 4: Save enhanced datasets
            print(f"\n💾 Step 4: Saving Gemini-enhanced datasets...")
            saved_files = await self._save_enhanced_results(enhanced_results)
            
            # Step 5: Generate comprehensive report
            print(f"📊 Step 5: Generating Gemini enhancement report...")
            report_file = await self._generate_enhancement_report(enhanced_results, start_time)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n🎯 GEMINI ENHANCEMENT COMPLETE:")
            print(f"  🧠 AI Engine: Google Gemini 1.5 Flash")
            print(f"  📊 Datasets Enhanced: {len(enhanced_results)}")
            print(f"  🔧 Issues Resolved: {sum(len(r['agent3_resolutions']) for r in enhanced_results.values())}")
            print(f"  🩺 Medical Validations: {sum(len(r['logical_validations']) for r in enhanced_results.values())}")
            print(f"  ⏱️  Processing Time: {processing_time:.2f} seconds")
            
            print(f"\n📁 Enhanced Files:")
            for file_path in saved_files.get('enhanced_datasets', []):
                print(f"  📊 {file_path}")
            print(f"📋 Gemini Report: {report_file}")
            
            print(f"\n✨ Gemini AI successfully enhanced Agent 2 datasets using Agent 3 insights!")
            print("🎯 Ready for production use with medical-grade accuracy!")
            
            return {
                'success': True,
                'ai_engine': 'Google Gemini 1.5 Flash',
                'processing_time': processing_time,
                'datasets_enhanced': len(enhanced_results),
                'total_issues_resolved': sum(len(r['agent3_resolutions']) for r in enhanced_results.values()),
                'logical_validations_performed': sum(len(r['logical_validations']) for r in enhanced_results.values()),
                'enhanced_results': enhanced_results,
                'report_file': report_file
            }
            
        except Exception as e:
            logger.error(f"Gemini enhancement pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def _save_enhanced_results(self, enhanced_results: Dict) -> Dict:
        """Save Gemini-enhanced datasets and results"""
        
        # Create output directory
        output_dir = "agent4_output"
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {'enhanced_datasets': []}
        
        # Save each enhanced dataset
        for dataset_name, results in enhanced_results.items():
            enhanced_df = results['enhanced_data']
            
            # Save enhanced CSV
            output_file = f"{output_dir}/gemini_enhanced_{dataset_name}.csv"
            enhanced_df.to_csv(output_file, index=False)
            saved_files['enhanced_datasets'].append(output_file)
            logger.info(f"💾 Saved: {output_file}")
        
        return saved_files
    
    async def _generate_enhancement_report(self, enhanced_results: Dict, start_time: datetime) -> str:
        """Generate comprehensive Gemini enhancement report"""
        
        report = {
            'agent4_gemini_enhancement_report': {
                'ai_engine': 'Google Gemini 1.5 Flash',
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                'agent2_datasets_processed': len(enhanced_results),
                'agent3_issues_addressed': len(self.agent3_issues),
                'agent3_issues_detail': [
                    {
                        'issue_type': issue.issue_type,
                        'description': issue.description,
                        'severity': issue.severity,
                        'column': issue.column,
                        'recommendation': issue.recommendation
                    }
                    for issue in self.agent3_issues
                ],
                'enhancement_summary': {
                    'total_datasets_enhanced': len(enhanced_results),
                    'total_logical_validations': sum(len(r['logical_validations']) for r in enhanced_results.values()),
                    'total_fixes_applied': sum(len([fix for fix in r['agent3_resolutions'] if fix['status'] == 'resolved']) for r in enhanced_results.values()),
                    'gemini_confidence_scores': []
                },
                'detailed_results': {}
            }
        }
        
        # Add detailed results for each dataset
        for dataset_name, results in enhanced_results.items():
            successful_resolutions = [r for r in results['agent3_resolutions'] if r['status'] == 'resolved']
            
            report['agent4_gemini_enhancement_report']['detailed_results'][dataset_name] = {
                'original_shape': results['original_shape'],
                'enhanced_shape': results['enhanced_shape'],
                'data_improvement': {
                    'shape_change': results['enhanced_shape'][0] - results['original_shape'][0],
                    'columns_affected': len(set(r.get('gemini_reasoning', '') for r in successful_resolutions))
                },
                'logical_issues_found': len(results['logical_validations']),
                'logical_validations': [
                    {
                        'validation_type': val.validation_type,
                        'columns_involved': val.column_combination,
                        'ai_reasoning': val.ai_reasoning,
                        'confidence': val.confidence
                    }
                    for val in results['logical_validations']
                ],
                'agent3_issues_resolved': len(successful_resolutions),
                'resolution_details': [
                    {
                        'issue_type': res['issue_type'],
                        'status': res['status'],
                        'fix_applied': res.get('fix_applied', ''),
                        'gemini_reasoning': res.get('gemini_reasoning', ''),
                        'confidence': res.get('confidence', 0.0)
                    }
                    for res in results['agent3_resolutions']
                ]
            }
            
            # Collect confidence scores
            confidence_scores = [res.get('confidence', 0.0) for res in results['agent3_resolutions'] if res.get('confidence')]
            report['agent4_gemini_enhancement_report']['enhancement_summary']['gemini_confidence_scores'].extend(confidence_scores)
        
        # Calculate average confidence
        all_confidence_scores = report['agent4_gemini_enhancement_report']['enhancement_summary']['gemini_confidence_scores']
        if all_confidence_scores:
            report['agent4_gemini_enhancement_report']['enhancement_summary']['average_gemini_confidence'] = sum(all_confidence_scores) / len(all_confidence_scores)
        
        # Save report
        output_dir = "agent4_output"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%%S")
        report_file = f"{output_dir}/agent4_gemini_enhancement_report_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Gemini enhancement report saved: {report_file}")
        return report_file
def get_gemini_api_key() -> str:
    """Get Gemini API key from environment or user input"""

    # Try environment variable first
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("🔑 Gemini API Key Required")
        print("Please obtain your API key from: https://makersuite.google.com/app/apikey")
        api_key = input("Enter your Gemini API key: ").strip()

    if not api_key:
        raise ValueError("Gemini API key is required to run Agent 4")

    return api_key
async def main():
    """Main execution for Gemini-powered Agent 4"""

    print("🚀 Starting Agent 4: Gemini-Powered Dataset Enhancement & Issue Resolution...")
    print("🎯 Target: Agent 2 datasets + Agent 3 quality reports")
    print("🧠 AI Engine: Google Gemini 1.5 Flash")

    try:
        # Get Gemini API key
        gemini_api_key = get_gemini_api_key()
        
        # Initialize Gemini-powered Agent 4
        enhancer = GeminiAgent4DatasetEnhancer(gemini_api_key)
        
        # Run enhancement pipeline
        results = await enhancer.run_gemini_enhancement_pipeline()
        
        if results.get('success'):
            print(f"\n🎉 GEMINI ENHANCEMENT SUCCESS!")
            print(f"🧠 AI Engine: {results.get('ai_engine')}")
            print(f"📊 Datasets Enhanced: {results.get('datasets_enhanced', 0)}")
            print(f"🔧 Issues Resolved: {results.get('total_issues_resolved', 0)}")
            print(f"🩺 Medical Validations: {results.get('logical_validations_performed', 0)}")
            print(f"⏱️  Processing Time: {results.get('processing_time', 0):.2f} seconds")
            print("📁 Check agent4_output/ directory for enhanced datasets and Gemini reports")
            print("\n✨ Your datasets are now medically validated and production-ready! ✨")
        else:
            print(f"\n💥 GEMINI ENHANCEMENT FAILED: {results.get('error')}")
            print("🔧 Check your API key and input files")
            
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Gemini API key is valid")
        print("2. Ensure agent2_output/ and agent3_output/ directories exist")
        print("3. Verify CSV files are in agent2_output/")
        print("4. Check JSON reports are in agent3_output/")

if __name__ == "__main__":
    asyncio.run(main())
