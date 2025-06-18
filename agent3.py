"""
Agent 3: AI-Driven Dataset Analysis & Quality Assessment Agent
Analyzes Agent 2 output, detects errors, and provides intelligent feedback
No hardcoded domain logic - fully AI-driven analysis
"""

import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import re
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and pandas objects"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif pd.isna(obj):
            return None
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class QualityIssue:
    """Represents a data quality issue found by AI analysis"""
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'completeness', 'consistency', 'validity', 'accuracy', 'uniqueness'
    description: str
    column: Optional[str]
    affected_records: int
    percentage_affected: float
    ai_recommendation: str
    fix_priority: int  # 1-5, 1 being highest
    example_values: List[Any]
    
    def to_dict(self):
        return convert_numpy_types(asdict(self))

@dataclass
class DatasetAnalysis:
    """Complete analysis of one dataset"""
    dataset_name: str
    file_path: str
    basic_stats: Dict[str, Any]
    quality_issues: List[QualityIssue]
    ai_assessment: Dict[str, Any]
    column_analysis: Dict[str, Any]
    data_patterns: Dict[str, Any]
    overall_quality_score: float
    ai_recommendations: List[str]
    error_summary: Dict[str, Any]
    
    def to_dict(self):
        return {
            'dataset_name': self.dataset_name,
            'file_path': self.file_path,
            'basic_stats': convert_numpy_types(self.basic_stats),
            'quality_issues': [issue.to_dict() for issue in self.quality_issues],
            'ai_assessment': convert_numpy_types(self.ai_assessment),
            'column_analysis': convert_numpy_types(self.column_analysis),
            'data_patterns': convert_numpy_types(self.data_patterns),
            'overall_quality_score': float(self.overall_quality_score),
            'ai_recommendations': self.ai_recommendations,
            'error_summary': convert_numpy_types(self.error_summary)
        }

@dataclass
class Agent3Output:
    """Complete Agent 3 analysis output"""
    original_context: Dict[str, Any]
    agent2_context: Dict[str, Any]
    dataset_analyses: List[DatasetAnalysis]
    cross_dataset_analysis: Dict[str, Any]
    overall_assessment: Dict[str, Any]
    critical_issues: List[QualityIssue]
    ai_feedback: Dict[str, Any]
    recommendations_for_agent4: List[str]
    processing_stats: Dict[str, Any]
    timestamp: str
    
    def to_dict(self):
        return {
            'original_context': convert_numpy_types(self.original_context),
            'agent2_context': convert_numpy_types(self.agent2_context),
            'dataset_analyses': [analysis.to_dict() for analysis in self.dataset_analyses],
            'cross_dataset_analysis': convert_numpy_types(self.cross_dataset_analysis),
            'overall_assessment': convert_numpy_types(self.overall_assessment),
            'critical_issues': [issue.to_dict() for issue in self.critical_issues],
            'ai_feedback': convert_numpy_types(self.ai_feedback),
            'recommendations_for_agent4': self.recommendations_for_agent4,
            'processing_stats': convert_numpy_types(self.processing_stats),
            'timestamp': self.timestamp
        }

class AIQualityAnalyzer:
    """
    AI-Driven Quality Analyzer - No hardcoded rules
    Uses AI to understand data quality issues contextually
    """
    
    def __init__(self, gemini_api_key: str, output_dir: str = "agent3_output"):
        self.gemini_api_key = gemini_api_key
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=gemini_api_key,
            model="gemini-1.5-flash",
            temperature=0.3  # Lower temperature for more consistent analysis
        )
        
        logger.info("AI-Driven Quality Analyzer initialized")

    async def analyze_agent2_output(self, agent2_output_path: str) -> Agent3Output:
        """Main analysis function - fully AI-driven"""
        start_time = datetime.now()
        
        try:
            # Load Agent 2 output
            with open(agent2_output_path, 'r') as f:
                agent2_data = json.load(f)
            
            logger.info("Starting AI-driven quality analysis of Agent 2 output")
            
            # Extract contexts
            original_context = agent2_data.get('original_context', {})
            agent2_context = {
                'generation_metadata': agent2_data.get('generation_metadata', {}),
                'test_cases': agent2_data.get('test_cases', []),
                'csv_files': agent2_data.get('csv_files', [])
            }
            
            # Analyze each dataset
            dataset_analyses = []
            for dataset_info in agent2_data.get('generated_datasets', []):
                analysis = await self._analyze_single_dataset(dataset_info, original_context)
                if analysis:
                    dataset_analyses.append(analysis)
            
            # Cross-dataset analysis
            cross_dataset_analysis = await self._perform_cross_dataset_analysis(dataset_analyses, original_context)
            
            # Overall assessment using AI
            overall_assessment = await self._generate_overall_assessment(dataset_analyses, original_context, agent2_context)
            
            # Extract critical issues
            critical_issues = []
            for analysis in dataset_analyses:
                critical_issues.extend([issue for issue in analysis.quality_issues if issue.severity == 'critical'])
            
            # Generate AI feedback
            ai_feedback = await self._generate_ai_feedback(dataset_analyses, original_context, agent2_context)
            
            # Generate recommendations for Agent 4
            agent4_recommendations = await self._generate_agent4_recommendations(
                dataset_analyses, overall_assessment, ai_feedback
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = Agent3Output(
                original_context=original_context,
                agent2_context=agent2_context,
                dataset_analyses=dataset_analyses,
                cross_dataset_analysis=cross_dataset_analysis,
                overall_assessment=overall_assessment,
                critical_issues=critical_issues,
                ai_feedback=ai_feedback,
                recommendations_for_agent4=agent4_recommendations,
                processing_stats={
                    'processing_time_seconds': processing_time,
                    'datasets_analyzed': len(dataset_analyses),
                    'total_issues_found': sum(len(a.quality_issues) for a in dataset_analyses),
                    'critical_issues_count': len(critical_issues),
                    'analysis_method': 'ai_intelligent'
                },
                timestamp=datetime.now().isoformat()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI Quality Analysis: {e}")
            raise

    async def _analyze_single_dataset(self, dataset_info: Dict, original_context: Dict) -> Optional[DatasetAnalysis]:
        """Analyze a single dataset using AI intelligence"""
        
        try:
            dataset_name = dataset_info.get('name', 'Unknown')
            file_path = dataset_info.get('file_path', '')
            
            logger.info(f"Analyzing dataset: {dataset_name}")
            
            # Load the actual CSV data
            if not os.path.exists(file_path):
                logger.warning(f"CSV file not found: {file_path}")
                return None
            
            df = pd.read_csv(file_path)
            
            # Basic statistical analysis
            basic_stats = self._calculate_basic_stats(df)
            
            # AI-driven column analysis
            column_analysis = await self._ai_analyze_columns(df, dataset_name, original_context)
            
            # AI-driven pattern detection
            data_patterns = await self._ai_detect_patterns(df, dataset_name, original_context)
            
            # AI-driven quality issue detection
            quality_issues = await self._ai_detect_quality_issues(df, dataset_name, original_context, column_analysis)
            
            # AI assessment
            ai_assessment = await self._ai_assess_dataset_quality(df, dataset_name, original_context, quality_issues)
            
            # Calculate overall quality score
            domain = original_context.get('user_context', {}).get('domain', 'general')
            overall_quality_score = await self._calculate_quality_score(quality_issues, basic_stats, ai_assessment, domain)
            
            # Get AI recommendations
            ai_recommendations = await self._ai_generate_recommendations(df, dataset_name, quality_issues, original_context)
            
            # Generate error summary
            error_summary = self._generate_error_summary(quality_issues)
            
            analysis = DatasetAnalysis(
                dataset_name=dataset_name,
                file_path=file_path,
                basic_stats=basic_stats,
                quality_issues=quality_issues,
                ai_assessment=ai_assessment,
                column_analysis=column_analysis,
                data_patterns=data_patterns,
                overall_quality_score=overall_quality_score,
                ai_recommendations=ai_recommendations,
                error_summary=error_summary
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing dataset {dataset_info.get('name', 'Unknown')}: {e}")
            return None

    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistical information"""
        
        try:
            stats = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'missing_values_total': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
                'column_types': df.dtypes.astype(str).to_dict(),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'text_columns': len(df.select_dtypes(include=['object']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
            }
            
            # Per-column missing values
            missing_by_column = df.isnull().sum()
            stats['missing_by_column'] = {col: count for col, count in missing_by_column.items() if count > 0}
            
            return convert_numpy_types(stats)
            
        except Exception as e:
            logger.error(f"Error calculating basic stats: {e}")
            return {'error': str(e)}

    async def _ai_analyze_columns(self, df: pd.DataFrame, dataset_name: str, original_context: Dict) -> Dict[str, Any]:
        """AI-driven analysis of individual columns"""
        
        system_prompt = """You are an expert data analyst. Analyze the provided dataset columns and provide intelligent insights about each column's quality, appropriateness, and potential issues.

Consider:
1. Column names and their appropriateness for the domain
2. Data types and whether they make sense
3. Value ranges and distributions
4. Missing data patterns
5. Potential encoding or format issues
6. Domain-specific validation requirements

Return JSON with this structure:
{
    "column_name": {
        "analysis": "detailed analysis of this column",
        "data_type_appropriate": true/false,
        "value_range_analysis": "analysis of value ranges",
        "missing_data_pattern": "pattern of missing data",
        "potential_issues": ["issue1", "issue2"],
        "domain_relevance_score": 0.0-1.0,
        "recommendations": ["rec1", "rec2"]
    }
}"""
        
        # Prepare column summary for AI
        column_summary = {}
        for col in df.columns:
            col_data = df[col]
            column_summary[col] = {
                'dtype': str(col_data.dtype),
                'null_count': int(col_data.isnull().sum()),
                'unique_count': int(col_data.nunique()),
                'sample_values': col_data.dropna().head(5).tolist() if not col_data.empty else []
            }
            
            # Add statistics for numeric columns
            if col_data.dtype in ['int64', 'float64']:
                column_summary[col]['min'] = float(col_data.min()) if col_data.dtype == 'float64' else int(col_data.min())
                column_summary[col]['max'] = float(col_data.max()) if col_data.dtype == 'float64' else int(col_data.max())
                column_summary[col]['mean'] = float(col_data.mean())
        
        user_query = f"""
        Dataset: {dataset_name}
        Original Query Domain: {original_context.get('user_context', {}).get('domain', 'general')}
        Original Intent: {original_context.get('user_context', {}).get('original_query', '')}
        
        Column Summary:
        {json.dumps(column_summary, indent=2, default=str)}
        
        Analyze each column for quality, appropriateness, and potential issues in the context of the original research intent.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return convert_numpy_types(analysis)
            else:
                return {'error': 'No valid JSON found in AI response'}
                
        except Exception as e:
            logger.warning(f"AI column analysis failed: {e}")
            return {'error': str(e), 'method': 'ai_analysis_failed'}

    async def _ai_detect_patterns(self, df: pd.DataFrame, dataset_name: str, original_context: Dict) -> Dict[str, Any]:
        """AI-driven pattern detection in data"""
        
        system_prompt = """You are a data pattern recognition expert. Analyze the dataset and identify important patterns, anomalies, and relationships.

Look for:
1. Distribution patterns in numeric data
2. Categorical value patterns
3. Correlation patterns between columns
4. Temporal patterns (if date columns exist)
5. Outlier patterns
6. Missing data patterns
7. Business logic violations
8. Suspicious value combinations

Return JSON with this structure:
{
    "distribution_patterns": {
        "column_name": "pattern description"
    },
    "correlation_patterns": ["pattern1", "pattern2"],
    "anomaly_patterns": ["anomaly1", "anomaly2"],
    "missing_data_patterns": ["pattern1", "pattern2"],
    "business_logic_issues": ["issue1", "issue2"],
    "overall_data_health": "assessment of overall data health"
}"""
        
        # Prepare data summary for AI
        data_summary = {
            'numeric_columns': {},
            'categorical_columns': {},
            'correlation_info': {},
            'missing_patterns': {}
        }
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                data_summary['numeric_columns'][col] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'skewness': float(col_data.skew() if len(col_data) > 1 else 0)
                }
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(10)
            data_summary['categorical_columns'][col] = {
                'unique_count': len(df[col].unique()),
                'top_values': value_counts.to_dict()
            }
        
        # Missing data patterns
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 0:
                data_summary['missing_patterns'][col] = missing_pct
        
        user_query = f"""
        Dataset: {dataset_name}
        Domain: {original_context.get('user_context', {}).get('domain', 'general')}
        Original Query: {original_context.get('user_context', {}).get('original_query', '')}
        
        Data Summary:
        {json.dumps(data_summary, indent=2, default=str)}
        
        Total Records: {len(df)}
        Total Columns: {len(df.columns)}
        
        Identify patterns, anomalies, and potential issues in this dataset.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                patterns = json.loads(json_match.group())
                return convert_numpy_types(patterns)
            else:
                return {'error': 'No valid JSON found in AI response'}
                
        except Exception as e:
            logger.warning(f"AI pattern detection failed: {e}")
            return {'error': str(e)}

    async def _ai_detect_quality_issues(self, df: pd.DataFrame, dataset_name: str, original_context: Dict, column_analysis: Dict) -> List[QualityIssue]:
        """AI-driven quality issue detection"""
        
        system_prompt = """You are a data quality expert. Based on the dataset analysis, identify specific quality issues that need attention.

For each issue, provide:
1. Severity level (critical, high, medium, low)
2. Category (completeness, consistency, validity, accuracy, uniqueness)
3. Detailed description
4. Affected column (if applicable)
5. Estimated percentage of records affected
6. Recommended fix approach
7. Fix priority (1-5, 1 being highest)

Return JSON array with this structure:
[
    {
        "severity": "critical/high/medium/low",
        "category": "completeness/consistency/validity/accuracy/uniqueness",
        "description": "detailed description of the issue",
        "column": "column_name or null",
        "estimated_affected_percentage": 0.0-100.0,
        "recommendation": "suggested fix approach",
        "priority": 1-5,
        "example_problematic_values": ["val1", "val2"]
    }
]"""
        
        # Prepare quality analysis data
        quality_data = {
            'basic_stats': {
                'total_records': len(df),
                'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
            },
            'column_analysis': column_analysis,
            'problematic_patterns': []
        }
        
        # Find potential issues
        for col in df.columns:
            col_data = df[col]
            col_issues = []
            
            # High missing data
            missing_pct = (col_data.isnull().sum() / len(df)) * 100
            if missing_pct > 20:
                col_issues.append(f"High missing data: {missing_pct:.1f}%")
            
            # Low uniqueness for non-categorical data
            if col_data.dtype in ['int64', 'float64']:
                unique_pct = (col_data.nunique() / len(col_data.dropna())) * 100
                if unique_pct < 10:
                    col_issues.append(f"Low uniqueness: {unique_pct:.1f}%")
            
            # Outliers in numeric data
            if col_data.dtype in ['int64', 'float64'] and len(col_data.dropna()) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))]
                if len(outliers) > len(col_data) * 0.05:  # More than 5% outliers
                    col_issues.append(f"High outlier rate: {len(outliers)}/{len(col_data)}")
            
            if col_issues:
                quality_data['problematic_patterns'].append({
                    'column': col,
                    'issues': col_issues
                })
        
        user_query = f"""
        Dataset: {dataset_name}
        Domain: {original_context.get('user_context', {}).get('domain', 'general')}
        Research Intent: {original_context.get('user_context', {}).get('original_query', '')}
        
        Quality Analysis Data:
        {json.dumps(quality_data, indent=2, default=str)}
        
        Identify specific quality issues that could impact the research objectives.
        Consider domain-specific requirements and data usage patterns.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                issues_data = json.loads(json_match.group())
                
                quality_issues = []
                for issue_data in issues_data:
                    # Calculate actual affected records
                    column = issue_data.get('column')
                    estimated_pct = issue_data.get('estimated_affected_percentage', 0)
                    affected_records = int((estimated_pct / 100) * len(df))
                    
                    # Get example values
                    example_values = issue_data.get('example_problematic_values', [])
                    
                    issue = QualityIssue(
                        severity=issue_data.get('severity', 'medium'),
                        category=issue_data.get('category', 'general'),
                        description=issue_data.get('description', 'Quality issue detected'),
                        column=column,
                        affected_records=affected_records,
                        percentage_affected=estimated_pct,
                        ai_recommendation=issue_data.get('recommendation', 'Manual review required'),
                        fix_priority=issue_data.get('priority', 3),
                        example_values=example_values
                    )
                    quality_issues.append(issue)
                
                return quality_issues
                
            else:
                return []
                
        except Exception as e:
            logger.warning(f"AI quality issue detection failed: {e}")
            return []

    async def _get_domain_specific_thresholds(self, domain: str) -> Dict[str, float]:
        """Use AI to determine appropriate thresholds for the domain"""
        
        system_prompt = """You are a data quality expert. For the given domain, provide appropriate thresholds for data quality metrics.
        Return a JSON object with thresholds and multipliers.
        Example: {
            "missing_threshold": 10,
            "missing_multiplier": 0.5,
            "duplicate_threshold": 5,
            "duplicate_multiplier": 0.3
        }
        """
        
        user_query = f"Determine appropriate data quality thresholds for the {domain} domain."
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                thresholds = json.loads(json_match.group())
                return thresholds
            else:
                # Fallback to default thresholds
                return {
                    "missing_threshold": 10,
                    "missing_multiplier": 0.5,
                    "duplicate_threshold": 5,
                    "duplicate_multiplier": 0.3
                }
                
        except Exception as e:
            logger.warning(f"Error getting domain thresholds: {e}")
            return {
                "missing_threshold": 10,
                "missing_multiplier": 0.5,
                "duplicate_threshold": 5,
                "duplicate_multiplier": 0.3
            }
    
    async def _ai_assess_dataset_quality(self, df: pd.DataFrame, dataset_name: str, original_context: Dict, quality_issues: List[QualityIssue]) -> Dict[str, Any]:
        """AI assessment of overall dataset quality"""
        
        system_prompt = """You are a senior data scientist assessing dataset quality for research purposes.

Provide a comprehensive assessment including:
1. Overall quality score (0-100)
2. Fitness for intended research purpose
3. Major strengths of the dataset
4. Major weaknesses that need attention
5. Confidence level in the data reliability
6. Suitability for different types of analysis

Return JSON with this structure:
{
    "overall_quality_score": 0-100,
    "fitness_for_purpose": "high/medium/low",
    "reliability_confidence": 0.0-1.0,
    "major_strengths": ["strength1", "strength2"],
    "major_weaknesses": ["weakness1", "weakness2"],
    "analysis_suitability": {
        "descriptive_analysis": "suitable/limited/unsuitable",
        "statistical_analysis": "suitable/limited/unsuitable",
        "machine_learning": "suitable/limited/unsuitable",
        "visualization": "suitable/limited/unsuitable"
    },
    "recommended_next_steps": ["step1", "step2"],
    "overall_assessment": "comprehensive assessment text"
}"""
        
        # Prepare assessment data
        assessment_data = {
            'dataset_info': {
                'name': dataset_name,
                'records': len(df),
                'columns': len(df.columns),
                'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            'quality_issues_summary': {
                'total_issues': len(quality_issues),
                'critical_issues': len([q for q in quality_issues if q.severity == 'critical']),
                'high_issues': len([q for q in quality_issues if q.severity == 'high']),
                'categories_affected': list(set([q.category for q in quality_issues]))
            },
            'original_research_intent': original_context.get('user_context', {}).get('original_query', ''),
            'domain': original_context.get('user_context', {}).get('domain', 'general')
        }
        
        user_query = f"""
        Dataset Assessment Request:
        {json.dumps(assessment_data, indent=2, default=str)}
        
        Quality Issues Found: {len(quality_issues)}
        Critical Issues: {len([q for q in quality_issues if q.severity == 'critical'])}
        
        Assess this dataset's quality and fitness for the intended research purpose.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                assessment = json.loads(json_match.group())
                return convert_numpy_types(assessment)
            else:
                return {'error': 'No valid JSON found in AI response'}
                
        except Exception as e:
            logger.warning(f"AI dataset assessment failed: {e}")
            return {'error': str(e)}

    async def _ai_generate_recommendations(self, df: pd.DataFrame, dataset_name: str, quality_issues: List[QualityIssue], original_context: Dict) -> List[str]:
        """Generate AI-driven recommendations"""
        
        system_prompt = """You are a data improvement consultant. Based on the identified quality issues and research context, provide actionable recommendations for improving dataset quality and usability.

Focus on:
1. Immediate fixes for critical issues
2. Data cleaning and preprocessing steps
3. Additional data collection needs
4. Analysis approach modifications
5. Quality assurance measures

Return a JSON array of specific, actionable recommendations:
["recommendation1", "recommendation2", ...]"""
        
        issues_summary = []
        for issue in quality_issues:
            issues_summary.append({
                'severity': issue.severity,
                'category': issue.category,
                'description': issue.description,
                'column': issue.column,
                'recommendation': issue.ai_recommendation
            })
        
        user_query = f"""
        Dataset: {dataset_name}
        Research Context: {original_context.get('user_context', {}).get('original_query', '')}
        Domain: {original_context.get('user_context', {}).get('domain', 'general')}
        
        Quality Issues Identified:
        {json.dumps(issues_summary, indent=2, default=str)}
        
        Dataset Size: {len(df)} records × {len(df.columns)} columns
        
        Provide specific, actionable recommendations for improving this dataset's quality and research value.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
                return recommendations
            else:
                return ["Manual review of data quality issues required"]
                
        except Exception as e:
            logger.warning(f"AI recommendations generation failed: {e}")
            return ["Error generating recommendations - manual review required"]

    async def _get_severity_weights(self, domain: str) -> Dict[str, float]:
        """Use AI to determine appropriate severity weights for the domain"""
        
        system_prompt = """You are a data quality expert. For the given domain, provide appropriate severity weights for different issue types.
        Return a JSON object with severity levels as keys and numeric weights as values.
        Example: {"critical": 20, "high": 10, "medium": 5, "low": 2}
        """
        
        user_query = f"Determine appropriate severity weights for quality issues in the {domain} domain."
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                weights = json.loads(json_match.group())
                return weights
            else:
                # Fallback to default weights
                return {'critical': 20, 'high': 10, 'medium': 5, 'low': 2}
                
        except Exception as e:
            logger.warning(f"Error getting severity weights: {e}")
            return {'critical': 20, 'high': 10, 'medium': 5, 'low': 2}
    
    async def _calculate_quality_score(self, quality_issues: List[QualityIssue], basic_stats: Dict, ai_assessment: Dict, domain: str) -> float:
        """Calculate overall quality score based on issues and AI assessment"""
        
        try:
            # Start with base score
            base_score = 100.0
            
            # Get AI-determined severity weights
            severity_multiplier = await self._get_severity_weights(domain)
            
            # Deduct points for quality issues
            for issue in quality_issues:
                deduction = severity_multiplier.get(issue.severity, 5) * (issue.percentage_affected / 100)
                base_score -= deduction
            
            # Use AI to determine appropriate thresholds and deductions for missing data and duplicates
            thresholds = await self._get_domain_specific_thresholds(domain)
            
            # Consider basic stats
            missing_pct = basic_stats.get('missing_percentage', 0)
            duplicate_pct = basic_stats.get('duplicate_percentage', 0)
            
            # Apply AI-determined thresholds
            if missing_pct > thresholds.get('missing_threshold', 10):
                base_score -= (missing_pct - thresholds.get('missing_threshold', 10)) * thresholds.get('missing_multiplier', 0.5)
            
            if duplicate_pct > thresholds.get('duplicate_threshold', 5):
                base_score -= (duplicate_pct - thresholds.get('duplicate_threshold', 5)) * thresholds.get('duplicate_multiplier', 0.3)
            
            # Consider AI assessment if available
            ai_score = ai_assessment.get('overall_quality_score')
            if ai_score is not None and isinstance(ai_score, (int, float)):
                # Weighted average of calculated score and AI score
                base_score = (base_score * 0.6) + (float(ai_score) * 0.4)
            
            # Ensure score is between 0 and 100
            return max(0.0, min(100.0, base_score))
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            return 50.0  # Default middle score if calculation fails

    def _generate_error_summary(self, quality_issues: List[QualityIssue]) -> Dict[str, Any]:
        """Generate summary of errors by category and severity"""
        
        summary = {
            'total_issues': len(quality_issues),
            'by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'by_category': {},
            'most_critical_issues': [],
            'columns_with_issues': set(),
            'overall_risk_level': 'low'
        }
        
        for issue in quality_issues:
            # Count by severity
            summary['by_severity'][issue.severity] += 1
            
            # Count by category
            if issue.category not in summary['by_category']:
                summary['by_category'][issue.category] = 0
            summary['by_category'][issue.category] += 1
            
            # Track columns with issues
            if issue.column:
                summary['columns_with_issues'].add(issue.column)
            
            # Collect critical issues
            if issue.severity in ['critical', 'high']:
                summary['most_critical_issues'].append({
                    'description': issue.description,
                    'severity': issue.severity,
                    'column': issue.column,
                    'affected_percentage': issue.percentage_affected
                })
        
        # Convert set to list for JSON serialization
        summary['columns_with_issues'] = list(summary['columns_with_issues'])
        
        # Determine overall risk level
        if summary['by_severity']['critical'] > 0:
            summary['overall_risk_level'] = 'critical'
        elif summary['by_severity']['high'] > 2:
            summary['overall_risk_level'] = 'high'
        elif summary['by_severity']['medium'] > 5:
            summary['overall_risk_level'] = 'medium'
        
        return convert_numpy_types(summary)

    async def _perform_cross_dataset_analysis(self, dataset_analyses: List[DatasetAnalysis], original_context: Dict) -> Dict[str, Any]:
        """Perform cross-dataset analysis using AI"""
        
        if len(dataset_analyses) <= 1:
            return {'message': 'Cross-dataset analysis requires multiple datasets'}
        
        system_prompt = """You are a data integration expert. Analyze multiple datasets for consistency, compatibility, and potential integration issues.

Look for:
1. Schema consistency across datasets
2. Value range compatibility
3. Data format consistency
4. Potential merge/join opportunities
5. Conflicting information patterns
6. Complementary data patterns

Return JSON with this structure:
{
    "compatibility_score": 0.0-1.0,
    "schema_consistency": {
        "common_columns": ["col1", "col2"],
        "conflicting_schemas": ["issue1", "issue2"]
    },
    "integration_opportunities": ["opportunity1", "opportunity2"],
    "potential_conflicts": ["conflict1", "conflict2"],
    "recommended_integration_approach": "approach description",
    "data_quality_comparison": {
        "dataset1": "quality summary",
        "dataset2": "quality summary"
    }
}"""
        
        # Prepare cross-dataset summary
        datasets_summary = []
        for analysis in dataset_analyses:
            summary = {
                'name': analysis.dataset_name,
                'columns': list(analysis.basic_stats.get('column_types', {}).keys()),
                'column_types': analysis.basic_stats.get('column_types', {}),
                'quality_score': analysis.overall_quality_score,
                'record_count': analysis.basic_stats.get('total_records', 0),
                'major_issues': [issue.description for issue in analysis.quality_issues if issue.severity in ['critical', 'high']]
            }
            datasets_summary.append(summary)
        
        user_query = f"""
        Research Context: {original_context.get('user_context', {}).get('original_query', '')}
        Domain: {original_context.get('user_context', {}).get('domain', 'general')}
        
        Datasets to Analyze:
        {json.dumps(datasets_summary, indent=2, default=str)}
        
        Analyze these datasets for cross-dataset compatibility and integration potential.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return convert_numpy_types(analysis)
            else:
                return {'error': 'No valid JSON found in AI response'}
                
        except Exception as e:
            logger.warning(f"Cross-dataset analysis failed: {e}")
            return {'error': str(e)}

    async def _generate_overall_assessment(self, dataset_analyses: List[DatasetAnalysis], original_context: Dict, agent2_context: Dict) -> Dict[str, Any]:
        """Generate overall assessment of all datasets using AI"""
        
        system_prompt = """You are a senior research data consultant. Provide a comprehensive assessment of the dataset collection's fitness for the intended research purpose.

Include:
1. Overall readiness for research
2. Major gaps that need addressing
3. Confidence level in research conclusions
4. Dataset collection strengths and weaknesses
5. Risk assessment for research validity
6. Recommended research methodology adjustments

Return JSON with this structure:
{
    "research_readiness_score": 0-100,
    "confidence_level": 0.0-1.0,
    "major_strengths": ["strength1", "strength2"],
    "critical_gaps": ["gap1", "gap2"],
    "research_validity_risk": "low/medium/high/critical",
    "methodology_recommendations": ["rec1", "rec2"],
    "data_collection_assessment": "overall assessment text",
    "next_steps_priority": ["step1", "step2", "step3"]
}"""
        
        # Prepare overall summary
        overall_summary = {
            'original_research_intent': original_context.get('user_context', {}).get('original_query', ''),
            'domain': original_context.get('user_context', {}).get('domain', 'general'),
            'datasets_generated': len(dataset_analyses),
            'generation_metadata': agent2_context.get('generation_metadata', {}),
            'quality_summary': {
                'avg_quality_score': sum(a.overall_quality_score for a in dataset_analyses) / len(dataset_analyses) if dataset_analyses else 0,
                'total_records': sum(a.basic_stats.get('total_records', 0) for a in dataset_analyses),
                'total_critical_issues': sum(len([i for i in a.quality_issues if i.severity == 'critical']) for a in dataset_analyses),
                'datasets_with_high_quality': len([a for a in dataset_analyses if a.overall_quality_score > 75])
            }
        }
        
        user_query = f"""
        Overall Assessment Request:
        {json.dumps(overall_summary, indent=2, default=str)}
        
        Individual Dataset Quality Scores:
        {[{'name': a.dataset_name, 'score': a.overall_quality_score, 'critical_issues': len([i for i in a.quality_issues if i.severity == 'critical'])} for a in dataset_analyses]}
        
        Assess the overall readiness of this dataset collection for the intended research.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                assessment = json.loads(json_match.group())
                return convert_numpy_types(assessment)
            else:
                return {'error': 'No valid JSON found in AI response'}
                
        except Exception as e:
            logger.warning(f"Overall assessment generation failed: {e}")
            return {'error': str(e)}

    async def _generate_ai_feedback(self, dataset_analyses: List[DatasetAnalysis], original_context: Dict, agent2_context: Dict) -> Dict[str, Any]:
        """Generate comprehensive AI feedback on Agent 2's work"""
        
        system_prompt = """You are reviewing the work of an AI dataset generation system. Provide constructive feedback on the quality of generated datasets and suggest improvements.

Evaluate:
1. Dataset generation approach effectiveness
2. Domain relevance and appropriateness
3. Data realism and quality
4. Coverage of research requirements
5. Technical implementation quality
6. Areas for improvement

Return JSON with this structure:
{
    "generation_quality_assessment": 0-100,
    "strengths_of_generation": ["strength1", "strength2"],
    "areas_for_improvement": ["improvement1", "improvement2"],
    "domain_appropriateness_score": 0-100,
    "data_realism_score": 0-100,
    "technical_quality_score": 0-100,
    "coverage_completeness_score": 0-100,
    "specific_feedback": {
        "positive_aspects": ["aspect1", "aspect2"],
        "concerns": ["concern1", "concern2"],
        "suggestions": ["suggestion1", "suggestion2"]
    },
    "overall_feedback_summary": "comprehensive feedback text"
}"""
        
        # Prepare feedback data
        feedback_data = {
            'generation_approach': agent2_context.get('generation_metadata', {}).get('method', 'unknown'),
            'original_request': original_context.get('user_context', {}).get('original_query', ''),
            'datasets_analysis_summary': []
        }
        
        for analysis in dataset_analyses:
            dataset_feedback = {
                'name': analysis.dataset_name,
                'quality_score': analysis.overall_quality_score,
                'record_count': analysis.basic_stats.get('total_records', 0),
                'column_count': analysis.basic_stats.get('total_columns', 0),
                'critical_issues': len([i for i in analysis.quality_issues if i.severity == 'critical']),
                'ai_assessment': analysis.ai_assessment.get('fitness_for_purpose', 'unknown')
            }
            feedback_data['datasets_analysis_summary'].append(dataset_feedback)
        
        user_query = f"""
        Agent 2 Performance Review:
        {json.dumps(feedback_data, indent=2, default=str)}
        
        Provide comprehensive feedback on the dataset generation quality and suggest improvements for future generations.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                feedback = json.loads(json_match.group())
                return convert_numpy_types(feedback)
            else:
                return {'error': 'No valid JSON found in AI response'}
                
        except Exception as e:
            logger.warning(f"AI feedback generation failed: {e}")
            return {'error': str(e)}

    async def _generate_agent4_recommendations(self, dataset_analyses: List[DatasetAnalysis], overall_assessment: Dict, ai_feedback: Dict) -> List[str]:
        """Generate specific recommendations for Agent 4"""
        
        system_prompt = """You are preparing recommendations for the next agent in the pipeline (Agent 4). Based on the quality analysis results, provide specific, actionable recommendations for the next phase of processing.

Consider:
1. Data preprocessing requirements
2. Quality improvement priorities
3. Analysis approach recommendations
4. Risk mitigation strategies
5. Additional validation needs
6. Research methodology adjustments

Return a JSON array of specific recommendations:
["recommendation1", "recommendation2", ...]

Focus on actionable items that Agent 4 can implement."""
        
        # Prepare recommendations context
        rec_context = {
            'overall_quality_summary': {
                'avg_quality_score': sum(a.overall_quality_score for a in dataset_analyses) / len(dataset_analyses) if dataset_analyses else 0,
                'critical_issues_total': sum(len([i for i in a.quality_issues if i.severity == 'critical']) for a in dataset_analyses),
                'research_readiness': overall_assessment.get('research_readiness_score', 50)
            },
            'ai_feedback_summary': {
                'generation_quality': ai_feedback.get('generation_quality_assessment', 50),
                'main_concerns': ai_feedback.get('areas_for_improvement', []),
                'technical_quality': ai_feedback.get('technical_quality_score', 50)
            },
            'priority_issues': []
        }
        
        # Collect high-priority issues across all datasets
        for analysis in dataset_analyses:
            for issue in analysis.quality_issues:
                if issue.severity in ['critical', 'high'] and issue.fix_priority <= 2:
                    rec_context['priority_issues'].append({
                        'dataset': analysis.dataset_name,
                        'issue': issue.description,
                        'category': issue.category,
                        'recommendation': issue.ai_recommendation
                    })
        
        user_query = f"""
        Agent 4 Recommendations Context:
        {json.dumps(rec_context, indent=2, default=str)}
        
        Based on this analysis, provide specific recommendations for Agent 4 to improve dataset quality and research readiness.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
                return recommendations
            else:
                return ["Comprehensive data quality review required", "Manual validation of critical issues needed"]
                
        except Exception as e:
            logger.warning(f"Agent 4 recommendations generation failed: {e}")
            return ["Error generating recommendations - manual review required"]

    def save_analysis_report(self, analysis_result: Agent3Output) -> str:
        """Save detailed analysis report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"quality_analysis_report_{timestamp}.json")
        
        with open(report_file, 'w') as f:
            json.dump(analysis_result.to_dict(), f, indent=2, cls=CustomJSONEncoder)
        
        return report_file

    def export_for_agent4(self, analysis_result: Agent3Output) -> Dict[str, Any]:
        """Export analysis in format suitable for Agent 4"""
        
        return {
            'agent3_analysis': analysis_result.to_dict(),
            'action_items': {
                'critical_issues': [issue.to_dict() for issue in analysis_result.critical_issues],
                'recommendations': analysis_result.recommendations_for_agent4,
                'priority_datasets': [
                    analysis.dataset_name for analysis in analysis_result.dataset_analyses 
                    if analysis.overall_quality_score < 60
                ]
            },
            'metadata': {
                'analysis_timestamp': analysis_result.timestamp,
                'total_datasets_analyzed': len(analysis_result.dataset_analyses),
                'analysis_method': 'ai_intelligent',
                'agent_version': '3.0'
            }
        }


# Main execution function
async def main():
    """Main execution function for Agent 3"""
    
    # Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    AGENT2_OUTPUT_FILE = "agent2_output/agent2_ai_output.json"
    
    print("\nAgent 3: AI-Driven Dataset Analysis & Quality Assessment")
    print("=" * 60)
    
    try:
        # Initialize AI Quality Analyzer
        analyzer = AIQualityAnalyzer(GEMINI_API_KEY)
        logger.info("AI-Driven Quality Analyzer initialized")
        
        # Check if Agent 2 output exists
        if not os.path.exists(AGENT2_OUTPUT_FILE):
            raise FileNotFoundError(f"Agent 2 output file not found: {AGENT2_OUTPUT_FILE}")
        
        print(f"Analyzing Agent 2 output: {AGENT2_OUTPUT_FILE}")
        logger.info("Starting comprehensive AI quality analysis")
        
        # Perform AI-driven analysis
        analysis_result = await analyzer.analyze_agent2_output(AGENT2_OUTPUT_FILE)
        
        print(f"\nAI QUALITY ANALYSIS COMPLETED:")
        print(f"  Processing Time: {analysis_result.processing_stats['processing_time_seconds']:.2f} seconds")
        print(f"  Analysis Method: AI-Intelligent (No Hardcoding)")
        print(f"  Datasets Analyzed: {len(analysis_result.dataset_analyses)}")
        
        print(f"\nQUALITY ANALYSIS RESULTS:")
        
        # Print dataset-by-dataset results
        for i, analysis in enumerate(analysis_result.dataset_analyses, 1):
            print(f"\n{i}. Dataset: {analysis.dataset_name}")
            print(f"   Quality Score: {analysis.overall_quality_score:.1f}/100")
            print(f"   Total Issues: {len(analysis.quality_issues)}")
            print(f"   Critical Issues: {len([q for q in analysis.quality_issues if q.severity == 'critical'])}")
            print(f"   AI Assessment: {analysis.ai_assessment.get('fitness_for_purpose', 'N/A')}")
            print(f"   Records: {analysis.basic_stats.get('total_records', 0)}")
            
            # Show top issues
            critical_issues = [q for q in analysis.quality_issues if q.severity in ['critical', 'high']][:3]
            if critical_issues:
                print(f"   Top Issues:")
                for issue in critical_issues:
                    print(f"     • [{issue.severity.upper()}] {issue.description}")
        
        # Overall assessment
        overall = analysis_result.overall_assessment
        print(f"\nOVERALL ASSESSMENT:")
        print(f"  Research Readiness: {overall.get('research_readiness_score', 'N/A')}/100")
        print(f"  Confidence Level: {overall.get('confidence_level', 'N/A')}")
        print(f"  Risk Level: {overall.get('research_validity_risk', 'N/A')}")
        
        # AI Feedback
        ai_feedback = analysis_result.ai_feedback
        print(f"\nAI FEEDBACK ON AGENT 2:")
        print(f"  Generation Quality: {ai_feedback.get('generation_quality_assessment', 'N/A')}/100")
        print(f"  Domain Appropriateness: {ai_feedback.get('domain_appropriateness_score', 'N/A')}/100")
        print(f"  Data Realism: {ai_feedback.get('data_realism_score', 'N/A')}/100")
        
        # Recommendations for Agent 4
        print(f"\nRECOMMENDATIONS FOR AGENT 4:")
        for i, rec in enumerate(analysis_result.recommendations_for_agent4[:5], 1):
            print(f"  {i}. {rec}")
        
        # Save results
        report_file = analyzer.save_analysis_report(analysis_result)
        print(f"\n✓ Analysis report saved: {report_file}")
        
        # Export for Agent 4
        agent4_ready = analyzer.export_for_agent4(analysis_result)
        agent4_file = os.path.join(analyzer.output_dir, 'agent3_for_agent4.json')
        with open(agent4_file, 'w') as f:
            json.dump(agent4_ready, f, indent=2, cls=CustomJSONEncoder)
        
        print(f"✓ Agent 4 input file ready: {agent4_file}")
        print(f"✓ Critical issues flagged: {len(analysis_result.critical_issues)}")
        print(f"✓ Ready for Agent 4 processing!")
        
    except Exception as e:
        logger.error(f"Error in AI Quality Analysis: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Install required packages if not already installed
    try:
        import pandas as pd
        import numpy as np
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print("Please install required packages: pip install pandas numpy langchain-google-genai python-dotenv")
        exit(1)
    
    asyncio.run(main())
