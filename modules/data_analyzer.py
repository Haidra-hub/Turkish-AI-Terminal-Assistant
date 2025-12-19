"""
Data Analyzer Module
Comprehensive data analysis utilities for Turkish-AI-Terminal-Assistant
"""

import json
import csv
from typing import Any, Dict, List, Union, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import statistics


class DataAnalyzer:
    """Main data analysis class providing comprehensive analysis tools"""
    
    def __init__(self):
        """Initialize the DataAnalyzer"""
        self.data = None
        self.analysis_results = {}
    
    def load_data(self, data: Union[List, Dict, str]) -> None:
        """
        Load data for analysis
        
        Args:
            data: Data to analyze (list, dict, or JSON string)
        """
        if isinstance(data, str):
            try:
                self.data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided")
        else:
            self.data = data
    
    def get_basic_statistics(self, values: List[Union[int, float]]) -> Dict[str, float]:
        """
        Calculate basic statistical measures
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary containing statistical measures
        """
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(values)
        
        stats = {
            'count': n,
            'sum': sum(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values),
        }
        
        if n > 1:
            stats['stdev'] = statistics.stdev(values)
            stats['variance'] = statistics.variance(values)
        
        # Calculate quartiles
        q1_index = n // 4
        q3_index = 3 * n // 4
        stats['q1'] = sorted_values[q1_index]
        stats['q3'] = sorted_values[q3_index]
        stats['iqr'] = stats['q3'] - stats['q1']
        
        return stats
    
    def analyze_list(self, data: List) -> Dict[str, Any]:
        """
        Comprehensive analysis of a list
        
        Args:
            data: List to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'length': len(data),
            'type': 'list',
            'unique_elements': len(set(str(x) for x in data)),
            'empty': len(data) == 0,
        }
        
        # Count occurrences
        element_counts = Counter(data)
        analysis['most_common'] = dict(element_counts.most_common(5))
        
        # Numeric analysis if applicable
        numeric_values = [x for x in data if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if numeric_values:
            analysis['statistics'] = self.get_basic_statistics(numeric_values)
        
        return analysis
    
    def analyze_dict(self, data: Dict) -> Dict[str, Any]:
        """
        Comprehensive analysis of a dictionary
        
        Args:
            data: Dictionary to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'keys_count': len(data),
            'type': 'dictionary',
            'keys': list(data.keys()),
            'empty': len(data) == 0,
        }
        
        # Analyze values
        value_types = Counter(type(v).__name__ for v in data.values())
        analysis['value_types'] = dict(value_types)
        
        # Check for numeric values
        numeric_values = [v for v in data.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numeric_values:
            analysis['statistics'] = self.get_basic_statistics(numeric_values)
        
        return analysis
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text data
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing text analysis results
        """
        analysis = {
            'total_characters': len(text),
            'total_characters_no_spaces': len(text.replace(' ', '')),
            'words': len(text.split()),
            'lines': len(text.split('\n')),
            'sentences': len([s for s in text.split('.') if s.strip()]),
        }
        
        # Character frequency
        char_freq = Counter(text.lower())
        analysis['most_common_chars'] = dict(char_freq.most_common(10))
        
        # Word frequency
        words = text.lower().split()
        word_freq = Counter(words)
        analysis['most_common_words'] = dict(word_freq.most_common(10))
        
        # Turkish language specific analysis
        analysis['vowel_count'] = sum(1 for c in text.lower() if c in 'aeıioöuü')
        analysis['consonant_count'] = sum(1 for c in text.lower() if c.isalpha() and c not in 'aeıioöuü')
        
        return analysis
    
    def find_outliers(self, values: List[Union[int, float]], method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect outliers in numeric data
        
        Args:
            values: List of numeric values
            method: Detection method ('iqr' or 'zscore')
            
        Returns:
            Dictionary containing outlier information
        """
        if not values or len(values) < 4:
            return {'outliers': [], 'method': method}
        
        result = {'method': method}
        
        if method == 'iqr':
            stats = self.get_basic_statistics(values)
            lower_bound = stats['q1'] - 1.5 * stats['iqr']
            upper_bound = stats['q3'] + 1.5 * stats['iqr']
            outliers = [x for x in values if x < lower_bound or x > upper_bound]
            result['bounds'] = {'lower': lower_bound, 'upper': upper_bound}
        else:  # zscore
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
            outliers = [x for x in values if abs((x - mean) / stdev) > 3]
        
        result['outliers'] = outliers
        result['count'] = len(outliers)
        result['percentage'] = (len(outliers) / len(values) * 100) if values else 0
        
        return result
    
    def compare_distributions(self, dist1: List, dist2: List) -> Dict[str, Any]:
        """
        Compare two distributions
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Comparison results
        """
        numeric_dist1 = [x for x in dist1 if isinstance(x, (int, float)) and not isinstance(x, bool)]
        numeric_dist2 = [x for x in dist2 if isinstance(x, (int, float)) and not isinstance(x, bool)]
        
        if not numeric_dist1 or not numeric_dist2:
            return {'error': 'Insufficient numeric data for comparison'}
        
        stats1 = self.get_basic_statistics(numeric_dist1)
        stats2 = self.get_basic_statistics(numeric_dist2)
        
        return {
            'distribution1': stats1,
            'distribution2': stats2,
            'mean_difference': stats1['mean'] - stats2['mean'],
            'median_difference': stats1['median'] - stats2['median'],
        }
    
    def group_by(self, data: List[Dict], key: str) -> Dict[str, List]:
        """
        Group list of dictionaries by a key
        
        Args:
            data: List of dictionaries
            key: Key to group by
            
        Returns:
            Grouped data
        """
        grouped = defaultdict(list)
        for item in data:
            if isinstance(item, dict) and key in item:
                grouped[item[key]].append(item)
        return dict(grouped)
    
    def filter_data(self, data: List[Dict], conditions: Dict) -> List[Dict]:
        """
        Filter data based on conditions
        
        Args:
            data: List of dictionaries
            conditions: Dictionary of conditions {key: value}
            
        Returns:
            Filtered list
        """
        result = []
        for item in data:
            if isinstance(item, dict):
                if all(item.get(k) == v for k, v in conditions.items()):
                    result.append(item)
        return result
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary of loaded data
        
        Returns:
            Summary dictionary
        """
        if self.data is None:
            return {'error': 'No data loaded'}
        
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'data_type': type(self.data).__name__,
        }
        
        if isinstance(self.data, list):
            summary.update(self.analyze_list(self.data))
        elif isinstance(self.data, dict):
            summary.update(self.analyze_dict(self.data))
        elif isinstance(self.data, str):
            summary.update(self.analyze_text(self.data))
        
        return summary
    
    def export_analysis(self, format: str = 'json') -> Union[str, List[List]]:
        """
        Export analysis results
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Exported data in specified format
        """
        if format == 'json':
            return json.dumps(self.analysis_results, indent=2, ensure_ascii=False)
        elif format == 'csv':
            # Convert to CSV format
            if isinstance(self.analysis_results, dict):
                return [list(self.analysis_results.keys()), list(self.analysis_results.values())]
        
        return ""


class TimeSeriesAnalyzer:
    """Analyze time series data"""
    
    @staticmethod
    def detect_trend(values: List[Union[int, float]]) -> str:
        """
        Detect trend in time series data
        
        Args:
            values: List of numeric values
            
        Returns:
            Trend direction ('uptrend', 'downtrend', 'stable')
        """
        if len(values) < 2:
            return 'stable'
        
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])
        
        if second_half > first_half * 1.05:
            return 'uptrend'
        elif second_half < first_half * 0.95:
            return 'downtrend'
        return 'stable'
    
    @staticmethod
    def moving_average(values: List[Union[int, float]], window: int = 3) -> List[float]:
        """
        Calculate moving average
        
        Args:
            values: List of numeric values
            window: Window size for moving average
            
        Returns:
            List of moving averages
        """
        if window > len(values):
            window = len(values)
        
        moving_avgs = []
        for i in range(len(values) - window + 1):
            avg = statistics.mean(values[i:i + window])
            moving_avgs.append(avg)
        
        return moving_avgs


def analyze_data(data: Any, analysis_type: str = 'general') -> Dict[str, Any]:
    """
    Convenience function for quick data analysis
    
    Args:
        data: Data to analyze
        analysis_type: Type of analysis ('general', 'text', 'numeric')
        
    Returns:
        Analysis results
    """
    analyzer = DataAnalyzer()
    analyzer.load_data(data)
    
    if analysis_type == 'text' and isinstance(data, str):
        return analyzer.analyze_text(data)
    elif analysis_type == 'numeric' and isinstance(data, list):
        numeric_values = [x for x in data if isinstance(x, (int, float)) and not isinstance(x, bool)]
        return analyzer.get_basic_statistics(numeric_values)
    else:
        return analyzer.generate_summary()


if __name__ == "__main__":
    # Example usage
    analyzer = DataAnalyzer()
    
    # Test with sample data
    sample_data = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
    analyzer.load_data(sample_data)
    
    print("Basic Statistics:")
    print(json.dumps(analyzer.get_basic_statistics(sample_data), indent=2))
    
    print("\nOutlier Detection:")
    print(json.dumps(analyzer.find_outliers(sample_data), indent=2))
    
    # Text analysis
    sample_text = "Turkish AI Terminal Assistant - Yapay Zeka Terminali"
    text_analysis = analyzer.analyze_text(sample_text)
    print("\nText Analysis:")
    print(json.dumps(text_analysis, indent=2, ensure_ascii=False))
