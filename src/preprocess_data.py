import pandas as pd
import numpy as np

class AdvancedDataProcessor:
    """Enhanced Data Processing Engine"""
    
    def __init__(self, null_threshold=0.7):
        # Configuration based on provided missing rate
        self.null_threshold = null_threshold
        self.columns_to_drop = [
            'company_name', 'internal_industry', 'internal_market_cluster',
            'direct_available_space', 'direct_availability_proportion',
            'direct_internal_class_rent', 'direct_overall_rent',
            'sublet_available_space', 'sublet_availability_proportion',
            'sublet_internal_class_rent', 'sublet_overall_rent'
        ]
        
        # Business rules configuration
        self.business_rules = {
            'year': {'min': 2018, 'max': 2024},
            'quarter': {'allowed': ['Q1', 'Q2', 'Q3', 'Q4']},
            'zip': {'digits': 6},
            'CBD_suburban': {'allowed': ['CBD', 'Suburban']},
            'internal_class': {'allowed': ['A', 'O']},
            'availability_proportion': {'min': 0, 'max': 1}
        }

    def _merge_datasets(self, dfs: dict[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple DataFrames into one"""
        df = dfs['Leases.csv']
        df.merge(dfs['Major Market Occupancy Data-revised.csv'], on=['quarter', 'year', 'market'], how='left')
        df.merge(dfs['Price and Availability Data.csv'], on=['quarter', 'year', 'market'], how='left')
        df.merge(dfs['Unemployment.csv'], on=['quarter', 'year', 'state'], how='left')
        return df
    
    def _drop_high_null_cols(self, df):
        """Delete columns based on predefined missing rate"""
        existing_cols = [c for c in self.columns_to_drop if c in df.columns]
        return df.drop(columns=existing_cols)

    def _clean_categoricals(self, df):
        """Clean categorical data"""
        # Standardize categorical value formats
        df['quarter'] = df['quarter'].str.upper().str.strip()
        df['region'] = df['region'].str.title()
        df['state'] = df['state'].str.upper().str[:2]
        
        # Process special fields
        df['zip'] = df['zip'].astype(str).str.extract('(\d{6})')[0]  # Extract 6-digit zip code
        return df

    def _validate_numerics(self, df):
        """Validate numeric fields"""
        # Rent reasonability check
        rent_cols = [c for c in df.columns if 'rent' in c.lower()]
        for col in rent_cols:
            df[col] = df[col].mask(df[col] < 0, np.nan)
        
        # Area non-negative check
        area_cols = ['RBA', 'available_space', 'leasedSF']
        for col in area_cols:
            df[col] = df[col].mask(df[col] < 0, np.nan)
            
        return df

    def _enforce_business_rules(self, df):
        """Enforce business rules"""
        # Time dimension validation
        current_year = pd.Timestamp.now().year
        df['year'] = df['year'].clip(2018, current_year)
        
        # Availability rate calculation validation
        mask = ~df['RBA'].isnull() & (df['RBA'] != 0)
        df.loc[mask, 'availability_proportion'] = (
            df['available_space'] / df['RBA']
        ).clip(0, 1)
        
        # Class rent validation
        class_a_mask = df['internal_class'] == 'A'
        # df.loc[class_a_mask, 'internal_class_rent'] = df.loc[class_a_mask, 'internal_class_rent'].mask(
        #     df.loc[class_a_mask, 'internal_class_rent'] < df['overall_rent'], 
        #     df['overall_rent'] * 1.1  # Automatically correct to 10% above market price
        # )
        return df

    def process(self, df):
        """Complete processing workflow"""
        # Column filtering
        df = self._drop_high_null_cols(df)
        
        # Type cleaning
        df = self._clean_categoricals(df)
        
        # Numeric processing
        df = self._validate_numerics(df)
        
        # Business rules execution
        df = self._enforce_business_rules(df)
        
        # type conversion
        cat_cols = df.select_dtypes(include='object').columns
        df[cat_cols] = df[cat_cols].astype('category')
        
        # Generate quality report
        self._generate_quality_report(df)
        
        return df

    def _generate_quality_report(self, df):
        """Data quality report"""
        report = []
        
        # Basic statistics
        report.append(f"Data dimensions after processing: {df.shape}")
        report.append(f"Remaining columns: {list(df.columns)}")
        
        # Key field validation
        year_violation = df[(df['year'] < 2018) | (df['year'] > 2024)]
        report.append(f"Number of abnormal year records: {len(year_violation)}")
        
        zip_violation = df['zip'].str.len() <= 6
        report.append(f"Number of invalid zip code records: {zip_violation.sum()}")
        
        # Business metrics validation
        avail_prop_violation = df[(df['availability_proportion'] < 0) | (df['availability_proportion'] > 1)]
        report.append(f"Number of abnormal availability rate records: {len(avail_prop_violation)}")
        
        print("\nData Quality Report:")
        print("\n".join(report))

        with open('data_quality.log', 'w') as f:
            f.write("\n".join(report))

if __name__ == "__main__":
    raw_dfs = {file: pd.read_csv(f'../data/{file}') for file in ['Leases.csv', 'Major Market Occupancy Data-revised.csv', 'Price and Availability Data.csv', 'Unemployment.csv']}
    processor = AdvancedDataProcessor(null_threshold=0.7)
    processed_df = processor.process(raw_dfs)
    processed_df.to_csv("processed_data.csv", index=False)