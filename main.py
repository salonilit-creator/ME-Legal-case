import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from data_preprocessing import LegalDataPreprocessor
from legal_case_classifier import LegalCaseClassifier
import sys
import warnings
warnings.filterwarnings('ignore')

class LegalCaseMLPipeline:
    """Complete ML pipeline for legal case classification"""
    
    def __init__(self, dataset_path='legal_cases_extended.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.preprocessor = None
        self.models = {}
        self.best_model = None
        self.results = {}
        
    def load_data(self):
        """Load legal case dataset"""
        print("\n[1] LOADING LEGAL CASE DATASET")
        print("="*70)
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"✓ Successfully loaded {len(self.df)} legal cases")
            print(f"✓ Features: {', '.join(self.df.columns.tolist())}")
            print(f"\nDataset Shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"✗ Error: {self.dataset_path} not found!")
            print("Please ensure the CSV file is in the project directory")
            return False
    
    def explore_data(self):
        """Explore and visualize dataset"""
        print("\n[2] DATA EXPLORATION")
        print("="*70)
        
        # Case type distribution
        print("\nCase Type Distribution:")
        case_type_counts = self.df['case_type'].value_counts()
        for case_type, count in case_type_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  • {case_type.upper():25s}: {count:3d} cases ({percentage:5.1f}%)")
        
        # Date range
        print(f"\nDate Range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        # Judges and Courts
        print(f"\nUnique Judges: {self.df['judge'].nunique()}")
        print(f"Unique Courts: {self.df['court'].nunique()}")
        
        # Sample records
        print("\nSample Records:")
        print(self.df[['case_id', 'case_type', 'judge', 'verdict']].head(5).to_string(index=False))
        
        # Verdict distribution
        print("\nVerdict Distribution:")
        verdict_counts = self.df['verdict'].value_counts()
        for verdict, count in verdict_counts.head(10).items():
            print(f"  • {verdict:30s}: {count:3d}")
    
    def preprocess_data(self):
        """Preprocess the dataset"""
        print("\n[3] DATA PREPROCESSING")
        print("="*70)
        
        self.preprocessor = LegalDataPreprocessor()
        
        print("\nCleaning legal case texts...")
        X_features, y_labels = self.preprocessor.preprocess_data(
            self.df.copy(),
            text_column='case_text',
            label_column='case_type'
        )
        
        print(f"✓ Text features processed")
        print(f"  - Feature matrix shape: {X_features.shape}")
        print(f"  - Number of features: {X_features.shape[1]}")
        print(f"  - Sparsity: {1.0 - (X_features.nnz / (X_features.shape[0] * X_features.shape[1])):.2%}")
        
        print(f"\n✓ Labels encoded")
        print(f"  - Number of classes: {len(np.unique(y_labels))}")
        print(f"  - Label mapping: {self.preprocessor.get_label_mapping()}")
        
        return X_features, y_labels
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        print("\n[4] DATA SPLITTING")
        print("="*70)
        
        # First split: train and temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=42
        )
        
        # Second split: val and test
        val_test_ratio = test_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_test_ratio, random_state=42
        )
        
        print(f"✓ Data split successfully:")
        print(f"  - Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  - Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"  - Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple classification models"""
        print("\n[5] MODEL TRAINING")
        print("="*70)
        
        models_to_train = [
            'random_forest',
            'gradient_boost',
            'logistic_regression'
        ]
        
        for model_type in models_to_train:
            print(f"\n{'─'*70}")
            print(f"Training: {model_type.upper()}")
            print(f"{'─'*70}")
            
            classifier = LegalCaseClassifier(model_type=model_type)
            val_accuracy = classifier.train(X_train, y_train, validation_split=0.2)
            
            self.models[model_type] = {
                'classifier': classifier,
                'val_accuracy': val_accuracy
            }
        
        print(f"\n{'─'*70}")
        print(f"✓ All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n[6] MODEL EVALUATION")
        print("="*70)
        
        best_accuracy = 0
        best_model_name = None
        
        evaluation_summary = []
        
        for model_name, model_data in self.models.items():
            print(f"\n{'─'*70}")
            print(f"Evaluating: {model_name.upper()}")
            print(f"{'─'*70}")
            
            classifier = model_data['classifier']
            metrics = classifier.evaluate(X_test, y_test)
            
            self.results[model_name] = metrics
            evaluation_summary.append({
                'Model': model_name.upper(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
            
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model_name = model_name
                self.best_model = classifier
        
        # Summary comparison
        print(f"\n{'─'*70}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'─'*70}")
        comparison_df = pd.DataFrame(evaluation_summary)
        print(comparison_df.to_string(index=False))
        
        print(f"\n{'✓'} BEST MODEL: {best_model_name.upper()} (Accuracy: {best_accuracy:.4f})")
        
        return best_model_name
    
    def make_predictions(self, X_test, y_test):
        """Make predictions on test set"""
        print("\n[7] MAKING PREDICTIONS")
        print("="*70)
        
        predictions = self.best_model.predict(X_test)
        
        # Create results dataframe
        label_to_case_type = {
            v: k for k, v in self.preprocessor.get_label_mapping().items()
        }
        
        results_df = pd.DataFrame({
            'actual_label': y_test,
            'predicted_label': predictions,
            'correct': y_test == predictions
        })
        
        results_df['actual_case_type'] = results_df['actual_label'].map(label_to_case_type)
        results_df['predicted_case_type'] = results_df['predicted_label'].map(label_to_case_type)
        
        accuracy = (results_df['correct'].sum() / len(results_df)) * 100
        print(f"\n✓ Prediction Accuracy on Test Set: {accuracy:.2f}%")
        
        print("\nSample Predictions (First 15):")
        sample_cols = ['actual_case_type', 'predicted_case_type', 'correct']
        print(results_df[sample_cols].head(15).to_string(index=False))
        
        # Prediction accuracy by case type
        print("\nPrediction Accuracy by Case Type:")
        for case_type in sorted(results_df['actual_case_type'].unique()):
            subset = results_df[results_df['actual_case_type'] == case_type]
            case_accuracy = (subset['correct'].sum() / len(subset)) * 100
            print(f"  • {case_type.upper():25s}: {case_accuracy:6.2f}% ({subset['correct'].sum()}/{len(subset)})")
        
        # Save results
        results_df.to_csv('prediction_results.csv', index=False)
        print(f"\n✓ Results saved to prediction_results.csv")
        
        return results_df
    
    def visualize_results(self, X_test, y_test, predictions):
        """Create visualization of results"""
        print("\n[8] CREATING VISUALIZATIONS")
        print("="*70)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Case Type Distribution
        ax1 = plt.subplot(2, 3, 1)
        case_counts = self.df['case_type'].value_counts()
        colors = plt.cm.Set3(range(len(case_counts)))
        case_counts.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Case Type Distribution in Dataset', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Case Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Model Performance Comparison
        ax2 = plt.subplot(2, 3, 2)
        models_list = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models_list]
        colors_models = ['#2ecc71', '#3498db', '#e74c3c']
        bars = ax2.bar(models_list, accuracies, color=colors_models)
        ax2.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.tick_params(axis='x', rotation=45)
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Confusion Matrix
        ax3 = plt.subplot(2, 3, 3)
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
        ax3.set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Predicted Label')
        ax3.set_ylabel('Actual Label')
        
        # 4. Metrics Comparison
        ax4 = plt.subplot(2, 3, 4)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        best_model_metrics = [
            self.results[list(self.results.keys())[0]]['accuracy'],
            self.results[list(self.results.keys())[0]]['precision'],
            self.results[list(self.results.keys())[0]]['recall'],
            self.results[list(self.results.keys())[0]]['f1_score']
        ]
        colors_metrics = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax4.bar(metrics_names, best_model_metrics, color=colors_metrics)
        ax4.set_title(f'Best Model Metrics', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim([0, 1])
        ax4.tick_params(axis='x', rotation=45)
        for bar, metric in zip(bars, best_model_metrics):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{metric:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Cases by Year
        ax5 = plt.subplot(2, 3, 5)
        cases_by_year = self.df['year'].value_counts().sort_index()
        ax5.plot(cases_by_year.index, cases_by_year.values, marker='o', linewidth=2, markersize=8, color='#2ecc71')
        ax5.set_title('Cases by Year', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Year')
        ax5.set_ylabel('Number of Cases')
        ax5.grid(True, alpha=0.3)
        
        # 6. Top Judges
        ax6 = plt.subplot(2, 3, 6)
        top_judges = self.df['judge'].value_counts().head(8)
        colors_judges = plt.cm.Set3(range(len(top_judges)))
        top_judges.plot(kind='barh', ax=ax6, color=colors_judges)
        ax6.set_title('Top 8 Judges by Case Count', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Number of Cases')
        
        plt.tight_layout()
        plt.savefig('legal_case_classification_results.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to legal_case_classification_results.png")
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n[9] GENERATING COMPREHENSIVE REPORT")
        print("="*70)
        
        report = f"""
{'='*70}
LEGAL CASE CLASSIFICATION AND PREDICTION - COMPREHENSIVE REPORT
{'='*70}

1. DATASET SUMMARY
   ├─ Total Cases: {len(self.df)}
   ├─ Case Types: {self.df['case_type'].nunique()}
   ├─ Date Range: {self.df['date'].min()} to {self.df['date'].max()}
   ├─ Unique Judges: {self.df['judge'].nunique()}
   └─ Unique Courts: {self.df['court'].nunique()}

2. CASE TYPE DISTRIBUTION
"""
        for case_type, count in self.df['case_type'].value_counts().items():
            percentage = (count / len(self.df)) * 100
            report += f"   ├─ {case_type.upper():20s}: {count:3d} ({percentage:5.1f}%)\n"
        
        report += f"""
3. MODEL TRAINING RESULTS
"""
        for model_name, model_data in self.models.items():
            report += f"   ├─ {model_name.upper():20s}: Val Accuracy = {model_data['val_accuracy']:.4f}\n"
        
        best_model_name = list(self.models.keys())[0]
        for name in self.models.keys():
            if self.results[name]['accuracy'] > self.results[best_model_name]['accuracy']:
                best_model_name = name
        
        report += f"""
4. BEST MODEL PERFORMANCE
   ├─ Model: {best_model_name.upper()}
   ├─ Accuracy: {self.results[best_model_name]['accuracy']:.4f}
   ├─ Precision: {self.results[best_model_name]['precision']:.4f}
   ├─ Recall: {self.results[best_model_name]['recall']:.4f}
   └─ F1-Score: {self.results[best_model_name]['f1_score']:.4f}

5. MODEL COMPARISON
"""
        for model_name in sorted(self.results.keys()):
            metrics = self.results[model_name]
            report += f"   {model_name.upper():20s}: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}\n"
        
        report += f"""
6. KEY INSIGHTS
   ├─ Dataset contains {len(self.df)} legal cases
   ├─ Cases span {self.df['case_type'].nunique()} different categories
   ├─ Best performing model achieved {self.results[best_model_name]['accuracy']:.2%} accuracy
   ├─ Multi-model comparison shows varying performance
   └─ Classification pipeline successfully deployed

7. OUTPUT FILES GENERATED
   ├─ prediction_results.csv (detailed predictions)
   ├─ legal_case_classification_results.png (visualizations)
   ├─ legal_case_random_forest_model.pkl (trained model)
   ├─ legal_case_gradient_boost_model.pkl (trained model)
   └─ legal_case_logistic_regression_model.pkl (trained model)

{'='*70}
PIPELINE EXECUTION COMPLETED SUCCESSFULLY
{'='*70}
"
        
        print(report)
        
        # Save report to file
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        
        print("\n✓ Report saved to classification_report.txt")
    
    def run_pipeline(self):
        """Execute the complete pipeline"""
        print("\n" + "="*70)
        print("LEGAL CASE ML CLASSIFICATION PIPELINE")
        print("="*70)
        
        # Load data
        if not self.load_data():
            return False
        
        # Explore data
        self.explore_data()
        
        # Preprocess
        X, y = self.preprocess_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        best_model_name = self.evaluate_models(X_test, y_test)
        
        # Make predictions
        predictions_df = self.make_predictions(X_test, y_test)
        
        # Get predictions for visualization
        predictions = self.best_model.predict(X_test)
        
        # Visualize
        self.visualize_results(X_test, y_test, predictions)
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*70)
        print("✓ COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
        print("="*70 + "\n")

def main():
    """Main execution function"""
    try:
        pipeline = LegalCaseMLPipeline(dataset_path='legal_cases_extended.csv')
        pipeline.run_pipeline()
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()