"""
Agent 1: Dataset-Focused Web Search Agent using content extraction
Uses your reliable web content extraction method for finding datasets
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import requests
from faker import Faker
from bs4 import BeautifulSoup
try:
    from readability import Document
except ImportError:
    # Fallback if readability is not available
    class Document:
        def __init__(self, html):
            self.html = html
        def summary(self):
            return self.html
import csv
import time
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    name: str
    description: str
    url: str
    format: str
    size_info: str
    columns: List[str]
    sample_data: List[Dict[str, Any]]
    source_context: str
    relevance_score: float
    data_quality_score: float
    extracted_content: str  # Added to store web content

@dataclass
class UserContext:
    original_query: str
    domain: str
    intent: str
    key_entities: List[str]
    data_requirements: List[str]
    expected_columns: List[str]
    use_cases: List[str]

@dataclass
class Agent1DatasetOutput:
    user_context: UserContext
    retrieved_datasets: List[DatasetInfo]
    search_metadata: Dict[str, Any]
    processing_time: float
    timestamp: str

class Agent1WebContentSearcher:
    """
    Agent 1: Dataset-focused web search using your content extraction method
    """
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        
        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=gemini_api_key,
            model="gemini-2.0-flash",
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        # Search configuration
        self.search_url = "https://lite.duckduckgo.com/lite/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Dataset-focused search patterns
        self.dataset_search_patterns = [
            "{query} dataset",
            "{query} data download",
            "{query} csv data",
            "{query} open data",
            "{query} machine learning dataset",
            "site:kaggle.com {query}",
            "site:data.gov {query}",
            "site:github.com {query} dataset"
        ]
        
        logger.info("Dataset-focused Agent 1 with web content extraction initialized")

    def get_urls(self, query: str, max_results: int = 15) -> List[str]:
        """Search for URLs related to a dataset query using your method"""
        try:
            logger.info(f"Searching for: {query}")
            response = requests.post(self.search_url, data={"q": query}, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            links = []
            
            # Look for various link selectors that might work
            selectors = [
                "a.result-link",
                "a[href*='http']",
                ".result a",
                "a.result__a"
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    for a in elements[:max_results]:
                        href = a.get('href')
                        if href and href.startswith('http'):
                            links.append(href)
                    break
            
            # If no links found with selectors, try finding all links
            if not links:
                all_links = soup.find_all('a', href=True)
                for link in all_links[:max_results]:
                    href = link['href']
                    if href and href.startswith('http'):
                        links.append(href)
            
            logger.info(f"Found {len(links)} URLs for query: {query}")
            return links
            
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            return []

    def get_cleaned_content(self, urls: List[str]) -> List[Dict[str, str]]:
        """Fetch and extract main content from URLs using your method"""
        contents = []
        
        for i, url in enumerate(urls):
            try:
                logger.info(f"Fetching content from {url}")
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                
                # Use readability to extract main content
                doc = Document(response.text)
                cleaned_text = doc.summary()
                
                # Also get the title
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.string if soup.title else "No title"
                
                contents.append({
                    'url': url,
                    'title': title.strip(),
                    'content': cleaned_text,
                    'raw_text': BeautifulSoup(cleaned_text, 'html.parser').get_text()
                })
                
                logger.info(f"Successfully extracted content from {url}")
                
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                continue
        
        return contents

    async def analyze_user_context(self, query: str) -> UserContext:
        """Analyze user query to understand dataset requirements"""
        
        analysis_prompt = f"""
        Analyze this query for dataset requirements: "{query}"
        
        The user needs datasets that can be used to generate test cases for this scenario.
        Focus on understanding what type of data structure and fields would be most relevant.
        
        Provide analysis in this JSON format:
        {{
            "domain": "primary domain (medical/financial/technology/education/business/general)",
            "intent": "what the user wants to achieve with the data",
            "key_entities": ["entity1", "entity2", "entity3"],
            "data_requirements": ["what type of data columns/fields needed"],
            "expected_columns": ["column1", "column2", "column3"],
            "use_cases": ["use_case1", "use_case2", "use_case3"]
        }}
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert data analyst specializing in dataset requirements analysis."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            response_text = response.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
                
                return UserContext(
                    original_query=query,
                    domain=analysis_data.get('domain', 'general'),
                    intent=analysis_data.get('intent', ''),
                    key_entities=analysis_data.get('key_entities', []),
                    data_requirements=analysis_data.get('data_requirements', []),
                    expected_columns=analysis_data.get('expected_columns', []),
                    use_cases=analysis_data.get('use_cases', [])
                )
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Context analysis error: {e}")
            return self._fallback_context_analysis(query)

    def _fallback_context_analysis(self, query: str) -> UserContext:
        """Simple fallback context analysis"""
        query_lower = query.lower()
        
        # Domain detection
        domain_keywords = {
            'medical': ['health', 'medical', 'patient', 'disease', 'doctor', 'hospital', 'drug', 'treatment', 'liver', 'alcohol', 'clinical'],
            'financial': ['money', 'stock', 'price', 'market', 'finance', 'bank', 'investment', 'economic', 'trading'],
            'education': ['student', 'school', 'education', 'grade', 'academic', 'university', 'learning'],
            'technology': ['software', 'tech', 'computer', 'programming', 'ai', 'machine learning', 'data science'],
            'business': ['customer', 'sales', 'business', 'company', 'revenue', 'marketing', 'profit']
        }
        
        domain = 'general'
        for d, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domain = d
                break
        
        # Extract entities
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        entities = [word for word in words if word.lower() not in ['the', 'and', 'for', 'with', 'data', 'dataset']][:5]
        
        return UserContext(
            original_query=query,
            domain=domain,
            intent="find relevant datasets for test case generation",
            key_entities=entities,
            data_requirements=["structured tabular data", "relevant columns", "sample records"],
            expected_columns=[],
            use_cases=["test case generation", "data analysis", "model training"]
        )

    async def search_for_datasets(self, context: UserContext, max_datasets: int = 5) -> List[DatasetInfo]:
        """Search for datasets using web content extraction"""
        
        all_contents = []
        search_queries_used = []
        
        # Generate dataset-focused search queries
        base_query = self._clean_query_for_search(context.original_query)
        
        search_queries = [
            pattern.format(query=base_query) 
            for pattern in self.dataset_search_patterns[:4]  # Use first 4 patterns to avoid too many requests
        ]
        
        logger.info(f"Searching with {len(search_queries)} dataset-focused queries...")
        
        # Perform searches and extract content
        for i, search_query in enumerate(search_queries):
            try:
                logger.info(f"Search {i+1}/{len(search_queries)}: {search_query}")
                
                # Get URLs
                urls = self.get_urls(search_query, max_results=5)  # Fewer URLs per search
                
                if urls:
                    # Get content from URLs
                    contents = self.get_cleaned_content(urls)
                    all_contents.extend(contents)
                    search_queries_used.append(search_query)
                    
                    logger.info(f"Extracted content from {len(contents)} pages")
                
                # Add delay between searches
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Search error for query '{search_query}': {e}")
                continue
        
        logger.info(f"Total content extracted from {len(all_contents)} web pages")
        
        # Analyze content to identify datasets
        datasets = await self._analyze_content_for_datasets(all_contents, context, max_datasets)
        
        return datasets

    async def _analyze_content_for_datasets(self, contents: List[Dict], context: UserContext, max_datasets: int) -> List[DatasetInfo]:
        """Analyze web content to identify and extract dataset information"""
        datasets = []
        
        for content_item in contents:
            try:
                dataset_info = await self._extract_dataset_from_content(content_item, context)
                if dataset_info:
                    datasets.append(dataset_info)
                    logger.info(f"Found dataset: {dataset_info.name}")
                
                if len(datasets) >= max_datasets:
                    break
                    
            except Exception as e:
                logger.error(f"Error analyzing content from {content_item['url']}: {e}")
                continue
        
        # If we don't have enough datasets, create some synthetic ones based on the content
        if len(datasets) < max_datasets:
            synthetic_count = max_datasets - len(datasets)
            synthetic_datasets = await self._generate_datasets_from_content(contents, context, synthetic_count)
            datasets.extend(synthetic_datasets)
        
        # Sort by relevance and quality
        datasets.sort(key=lambda x: (x.relevance_score + x.data_quality_score) / 2, reverse=True)
        
        return datasets[:max_datasets]

    async def _extract_dataset_from_content(self, content_item: Dict, context: UserContext) -> Optional[DatasetInfo]:
        """Extract dataset information from web content"""
        
        url = content_item['url']
        title = content_item['title']
        raw_text = content_item['raw_text']
        html_content = content_item['content']
        
        # Check if this looks like a dataset page
        dataset_indicators = [
            'dataset', 'csv', 'download', 'data', 'columns', 'rows',
            'kaggle', 'github', 'data.gov', 'machine learning'
        ]
        
        text_lower = raw_text.lower()
        if not any(indicator in text_lower for indicator in dataset_indicators):
            return None
        
        try:
            # Use Gemini to analyze the content and extract dataset information
            analysis_prompt = f"""
            Analyze the following web page content to extract dataset information:
            
            URL: {url}
            Title: {title}
            Content: {raw_text[:3000]}...
            
            User is looking for: {context.original_query}
            Domain: {context.domain}
            
            If this page contains or describes a dataset, extract the information in this JSON format:
            {{
                "is_dataset": true/false,
                "dataset_name": "name of the dataset",
                "description": "description of what the dataset contains",
                "file_format": "csv/json/excel/other",
                "estimated_columns": ["col1", "col2", "col3"],
                "data_size": "size information if available",
                "download_available": true/false,
                "relevance_to_query": 0.0-1.0
            }}
            
            Only return the JSON, no other text.
            """
            
            messages = [
                SystemMessage(content="You are an expert at analyzing web content to identify datasets."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            response_text = response.content
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                if analysis.get('is_dataset', False):
                    # Create dataset info
                    dataset_info = DatasetInfo(
                        name=analysis.get('dataset_name', title),
                        description=analysis.get('description', f'Dataset related to {context.original_query}'),
                        url=url,
                        format=analysis.get('file_format', 'unknown'),
                        size_info=analysis.get('data_size', 'Size not specified'),
                        columns=analysis.get('estimated_columns', []),
                        sample_data=self._generate_sample_data_from_content(analysis.get('estimated_columns', []), context),
                        source_context=f"Web content from {url}",
                        relevance_score=analysis.get('relevance_to_query', 0.5),
                        data_quality_score=self._calculate_content_quality_score(content_item, analysis),
                        extracted_content=raw_text[:1000]  # Store first 1000 chars
                    )
                    
                    return dataset_info
            
        except Exception as e:
            logger.error(f"Error analyzing content with Gemini: {e}")
        
        return None

    def _generate_sample_data_from_content(self, columns: List[str], context: UserContext) -> List[Dict[str, Any]]:
        """Generate realistic sample data based on columns and context"""
        if not columns:
            return []
        
        sample_data = {}
        
        for col in columns:
            col_lower = col.lower()
            
            # Generate context-appropriate sample data
            if context.domain == 'medical':
                if 'age' in col_lower:
                    sample_data[col] = 45
                elif 'patient' in col_lower or 'id' in col_lower:
                    sample_data[col] = 'P001'
                elif 'gender' in col_lower or 'sex' in col_lower:
                    sample_data[col] = 'Male'
                elif 'alcohol' in col_lower:
                    sample_data[col] = 'Yes'
                elif 'liver' in col_lower:
                    sample_data[col] = 'Cirrhosis'
                else:
                    sample_data[col] = 'Sample Value'
            
            elif context.domain == 'financial':
                if 'price' in col_lower:
                    sample_data[col] = 100.50
                elif 'date' in col_lower:
                    sample_data[col] = '2024-01-01'
                elif 'symbol' in col_lower:
                    sample_data[col] = 'STOCK'
                else:
                    sample_data[col] = 'Sample Value'
            
            else:
                sample_data[col] = 'Sample Value'
        
        return [sample_data]

    def _calculate_content_quality_score(self, content_item: Dict, analysis: Dict) -> float:
        """Calculate quality score based on content analysis"""
        score = 0.5  # Base score
        
        # Content length bonus
        content_length = len(content_item['raw_text'])
        if content_length > 500:
            score += 0.1
        if content_length > 1000:
            score += 0.1
        
        # Columns information bonus
        if analysis.get('estimated_columns'):
            score += 0.15
        
        # Download availability bonus
        if analysis.get('download_available'):
            score += 0.15
        
        # Source reliability bonus
        reliable_sources = ['kaggle.com', 'github.com', 'data.gov', 'uci.edu']
        if any(source in content_item['url'].lower() for source in reliable_sources):
            score += 0.2
        
        return min(score, 1.0)

    async def _generate_datasets_from_content(self, contents: List[Dict], context: UserContext, count: int) -> List[DatasetInfo]:
        """Generate synthetic datasets based on the web content found"""
        datasets = []
        
        # Analyze all content to understand what kind of data is being discussed
        all_text = ' '.join([item['raw_text'] for item in contents[:3]])  # Use first 3 content items
        
        try:
            synthesis_prompt = f"""
            Based on the following web content about "{context.original_query}", generate {count} realistic dataset descriptions:
            
            Web Content Summary: {all_text[:2000]}...
            
            User Query: {context.original_query}
            Domain: {context.domain}
            
            Create {count} different datasets that would be relevant. For each dataset, provide:
            {{
                "datasets": [
                    {{
                        "name": "dataset name",
                        "description": "what the dataset contains",
                        "columns": ["col1", "col2", "col3"],
                        "format": "csv/json/excel"
                    }}
                ]
            }}
            
            Only return the JSON, no other text.
            """
            
            messages = [
                SystemMessage(content="You are an expert at creating realistic dataset descriptions based on research content."),
                HumanMessage(content=synthesis_prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            response_text = response.content
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                synthesis_data = json.loads(json_match.group())
                
                for i, dataset_data in enumerate(synthesis_data.get('datasets', [])[:count]):
                    dataset = DatasetInfo(
                        name=dataset_data.get('name', f'Synthesized Dataset {i+1}'),
                        description=dataset_data.get('description', f'Dataset relevant to {context.original_query}'),
                        url=f'synthetic_from_web_content_{i+1}',
                        format=dataset_data.get('format', 'csv'),
                        size_info='Estimated based on web research',
                        columns=dataset_data.get('columns', []),
                        sample_data=self._generate_sample_data_from_content(dataset_data.get('columns', []), context),
                        source_context=f'Synthesized from web research about {context.original_query}',
                        relevance_score=0.8,
                        data_quality_score=0.7,
                        extracted_content=f'Based on research: {all_text[:500]}...'
                    )
                    datasets.append(dataset)
        
        except Exception as e:
            logger.error(f"Error generating synthetic datasets: {e}")
            # Fallback: generate simple synthetic datasets
            for i in range(count):
                if context.domain == 'medical':
                    name = f"Medical Research Dataset {i + 1}"
                    description = f"Clinical dataset containing patient information relevant to: {context.original_query}"
                    columns = ['patient_id', 'age', 'gender', 'condition', 'treatment', 'outcome']
                elif context.domain == 'financial':
                    name = f"Financial Analysis Dataset {i + 1}"
                    description = f"Financial dataset with market data relevant to: {context.original_query}"
                    columns = ['date', 'symbol', 'price', 'volume', 'market_cap']
                else:
                    name = f"Research Dataset {i + 1}"
                    description = f"Structured dataset relevant to: {context.original_query}"
                    columns = ['id', 'category', 'value', 'status', 'date']
                
                dataset = DatasetInfo(
                    name=name,
                    description=description,
                    url=f'synthetic_fallback_{i+1}',
                    format='csv',
                    size_info=f'~{10+i*5}KB estimated',
                    columns=columns,
                    sample_data=self._generate_sample_data_from_content(columns, context),
                    source_context=f"Synthetic Dataset - Generated for {context.domain} domain",
                    relevance_score=0.7,
                    data_quality_score=0.75,
                    extracted_content='Fallback synthetic dataset'
                )
                datasets.append(dataset)
        
        return datasets

    def _clean_query_for_search(self, query: str) -> str:
        """Clean query for better search results"""
        # Remove common words that don't help in dataset search
        stop_words = ['data', 'dataset', 'information', 'analysis', 'study', 'research']
        words = query.split()
        cleaned_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(cleaned_words) if cleaned_words else query

    async def process_query(self, user_query: str, max_datasets: int = 5) -> Agent1DatasetOutput:
        """Main processing function for Agent 1"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing dataset search query: {user_query}")
            
            # Step 1: Analyze user context
            context = await self.analyze_user_context(user_query)
            logger.info(f"Identified domain: {context.domain}")
            
            # Step 2: Search for datasets using web content extraction
            datasets = await self.search_for_datasets(context, max_datasets)
            logger.info(f"Found {len(datasets)} relevant datasets")
            
            # Step 3: Create output
            processing_time = time.time() - start_time
            
            output = Agent1DatasetOutput(
                user_context=context,
                retrieved_datasets=datasets,
                search_metadata={
                    'total_datasets_found': len(datasets),
                    'domain_detected': context.domain,
                    'search_method': 'web_content_extraction',
                    'processing_time_seconds': processing_time
                },
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
            return output
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def export_for_agent2(self, output: Agent1DatasetOutput) -> Dict[str, Any]:
        """Export results in format suitable for Agent 2"""
        return {
            'user_context': {
                'original_query': output.user_context.original_query,
                'domain': output.user_context.domain,
                'intent': output.user_context.intent,
                'key_entities': output.user_context.key_entities,
                'expected_columns': output.user_context.expected_columns,
                'data_requirements': output.user_context.data_requirements
            },
            'available_datasets': [
                {
                    'name': dataset.name,
                    'description': dataset.description,
                    'url': dataset.url,
                    'format': dataset.format,
                    'columns': dataset.columns,
                    'sample_data': dataset.sample_data,
                    'source_context': dataset.source_context,
                    'relevance_score': dataset.relevance_score,
                    'quality_score': dataset.data_quality_score,
                    'extracted_content_preview': dataset.extracted_content[:200] + '...' if len(dataset.extracted_content) > 200 else dataset.extracted_content
                }
                for dataset in output.retrieved_datasets
            ],
            'recommendations': {
                'best_dataset': output.retrieved_datasets[0].name if output.retrieved_datasets else None,
                'total_available': len(output.retrieved_datasets),
                'domain': output.user_context.domain,
                'search_summary': f"Found {len(output.retrieved_datasets)} datasets using web content analysis in {output.processing_time:.1f}s"
            },
            'metadata': output.search_metadata,
            'web_research_summary': self._create_research_summary(output.retrieved_datasets)
        }

    def _create_research_summary(self, datasets: List[DatasetInfo]) -> str:
        """Create a summary of the web research findings"""
        if not datasets:
            return "No datasets found through web research."
        
        real_datasets = [d for d in datasets if not d.url.startswith('synthetic')]
        synthetic_datasets = [d for d in datasets if d.url.startswith('synthetic')]
        
        summary = f"Web research analysis:\n"
        summary += f"- Found {len(real_datasets)} real dataset sources\n"
        summary += f"- Generated {len(synthetic_datasets)} synthetic datasets based on research\n"
        
        if real_datasets:
            summary += f"- Top real dataset: {real_datasets[0].name}\n"
            summary += f"- Best relevance score: {max(d.relevance_score for d in real_datasets):.2f}\n"
        
        return summary
    async def generate_recommendations(self, context: UserContext, datasets: List[DatasetInfo]) -> Dict[str, Any]:
        """Generate recommendations based on retrieved datasets and user context"""
        logger.info("Generating recommendations...")
        recommendation_prompt = f"""
        Based on the user's query: {context.original_query}
        And the retrieved datasets:
        {json.dumps([asdict(d) for d in datasets], indent=2)}
        
        Provide recommendations for the next steps in dataset generation and enhancement.
        Focus on potential data schemas, quality checks, and enhancement opportunities.
        
        Return a JSON object with a 'recommendations' key containing a list of strings.
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert in data pipeline recommendations."),
                HumanMessage(content=recommendation_prompt)
            ]
            
            response = await asyncio.to_thread(self.llm.invoke, messages)
            response_text = response.content
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                recommendations_data = json.loads(json_match.group())
                return recommendations_data.get('recommendations', {})
            else:
                logger.warning("No JSON found in recommendation response, returning empty recommendations.")
                return {}
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {}

    def _clean_query_for_search(self, query: str) -> str:
        """Clean the query for web search"""
        # Remove special characters and limit length
        cleaned_query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        return cleaned_query.strip()[:100]

    def _generate_sample_data_from_content(self, columns: List[str], context: UserContext) -> List[Dict[str, Any]]:
        """Generate sample data based on columns and context"""
        sample_data = []
        if not columns:
            return sample_data

        # Simple Faker-based sample data generation
        fake = Faker()
        for _ in range(3):  # Generate 3 sample rows
            row = {}
            for col in columns:
                col_lower = col.lower()
                if 'id' in col_lower:
                    row[col] = fake.uuid4()
                elif 'name' in col_lower:
                    row[col] = fake.name()
                elif 'age' in col_lower:
                    row[col] = fake.random_int(min=18, max=90)
                elif 'date' in col_lower:
                    row[col] = fake.date_this_century().isoformat()
                elif 'email' in col_lower:
                    row[col] = fake.email()
                elif 'description' in col_lower or 'text' in col_lower:
                    row[col] = fake.sentence()
                elif 'value' in col_lower or 'amount' in col_lower:
                    row[col] = round(fake.random_number(digits=5) / 100, 2)
                else:
                    row[col] = fake.word()
            sample_data.append(row)
        return sample_data

    async def _generate_datasets_from_content(self, contents: List[Dict], context: UserContext, count: int) -> List[DatasetInfo]:
        """Generate synthetic datasets from content if not enough real ones are found"""
        synthetic_datasets = []
        if not contents:
            return synthetic_datasets

        # Use Gemini to create synthetic dataset info based on content and context
        for i in range(count):
            content_sample = contents[i % len(contents)]['raw_text'][:1000] # Use a sample of content
            synthetic_prompt = f"""
            Based on the following content and user query, generate a synthetic dataset description.
            Content sample: {content_sample}
            User query: {context.original_query}
            
            Provide information in this JSON format:
            {{
                "dataset_name": "Synthetic Dataset {i+1}",
                "description": "A synthetic dataset generated based on the user's query and web content.",
                "file_format": "csv",
                "estimated_columns": ["id", "data_point", "value"],
                "data_size": "small",
                "relevance_to_query": 0.7,
                "data_quality_score": 0.8
            }}
            """
            try:
                messages = [
                    SystemMessage(content="You are an AI assistant that generates synthetic dataset descriptions."),
                    HumanMessage(content=synthetic_prompt)
                ]
                response = await asyncio.to_thread(self.llm.invoke, messages)
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    synthetic_info = json.loads(json_match.group())
                    synthetic_datasets.append(DatasetInfo(
                        name=synthetic_info.get('dataset_name', f'Synthetic Dataset {i+1}'),
                        description=synthetic_info.get('description', 'A synthetic dataset.'),
                        url='synthetic_data',
                        format=synthetic_info.get('file_format', 'csv'),
                        size_info=synthetic_info.get('data_size', 'small'),
                        columns=synthetic_info.get('estimated_columns', ['id', 'value']),
                        sample_data=self._generate_sample_data_from_content(synthetic_info.get('estimated_columns', ['id', 'value']), context),
                        source_context='Generated synthetically',
                        relevance_score=synthetic_info.get('relevance_to_query', 0.7),
                        data_quality_score=synthetic_info.get('data_quality_score', 0.8),
                        extracted_content=''
                    ))
            except Exception as e:
                logger.error(f"Error generating synthetic dataset: {e}")
        return synthetic_datasets




# Example usage function
async def main():
    """Example usage of Agent 1 with web content extraction"""
    
    # Initialize Agent 1
    gemini_api_key = "Gemini_API_KEY"  # Replace with your actual API key
    agent1 = Agent1WebContentSearcher(gemini_api_key)
    
    # Test 
    test_query = input("Enter your query: ")
    # test_query = "effect of alcohol on patients with liver problems"
    
    try:
        print(f"Searching for datasets related to: {test_query}")
        print("=" * 60)
        
        # Process the query
        result = await agent1.process_query(test_query, max_datasets=3)
        
        # Display results
        print(f"\nCONTEXT ANALYSIS:")
        print(f"  Query: {result.user_context.original_query}")
        print(f"  Domain: {result.user_context.domain}")
        print(f"  Intent: {result.user_context.intent}")
        print(f"  Key Entities: {', '.join(result.user_context.key_entities)}")
        
        print(f"\nFOUND DATASETS:")
        for i, dataset in enumerate(result.retrieved_datasets, 1):
            print(f"\n{i}. {dataset.name}")
            print(f"   URL: {dataset.url}")
            print(f"   Description: {dataset.description}")
            print(f"   Format: {dataset.format}")
            print(f"   Relevance: {dataset.relevance_score:.2f}")
            print(f"   Quality: {dataset.data_quality_score:.2f}")
            print(f"   Source: {dataset.source_context}")
            if dataset.columns:
                print(f"   Columns ({len(dataset.columns)}): {', '.join(dataset.columns[:5])}...")
            if dataset.sample_data:
                print(f"   Sample: {dataset.sample_data[0]}")
            if dataset.extracted_content and not dataset.url.startswith('synthetic'):
                print(f"   Content Preview: {dataset.extracted_content[:100]}...")
        
        # Export for Agent 2
        agent2_ready = agent1.export_for_agent2(result)
        print(f"\nREADY FOR AGENT 2:")
        print(f"  User Context: {agent2_ready['user_context']['original_query']}")
        print(f"  Available Datasets: {agent2_ready['recommendations']['total_available']}")
        print(f"  Best Dataset: {agent2_ready['recommendations']['best_dataset']}")
        print(f"  Web Research Summary: {agent2_ready['web_research_summary']}")
        
        # Save to file for Agent 2
        with open('agent1_output.json', 'w') as f:
            json.dump(agent2_ready, f, indent=2, default=str)
        print(f"  Output saved to: agent1_output.json")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    import asyncio
    
    # Install required packages if not already installed
    try:
        import readability
        from bs4 import BeautifulSoup
    except ImportError:
        exit(1)
    
    asyncio.run(main())

    