"""
Credit Risk Scoring System with Gradient Boosting and SHAP Explanations
================================================================================
This system implements a credit risk scoring model with:
- Gradient Boosting with monotonic constraints
- Benchmarking against Logistic Regression and Random Forest
- SHAP-based reason codes for interpretability
- Interactive Gradio interface

Dataset: German Credit Data from UCI ML Repository
Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import lightgbm as lgb
import shap
import gradio as gr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CreditRiskScorer:
    """Credit Risk Scoring System with multiple models and SHAP explanations"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.shap_explainers = {}
        self.evaluation_results = {}
        
    def load_german_credit_data(self, filepath):
        """
        Load and preprocess German Credit Data
        
        Args:
            filepath: Path to german.data file
        """
        # Column names for German Credit Data
        columns = [
            'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
            'savings_status', 'employment', 'installment_rate', 'personal_status',
            'other_parties', 'residence_since', 'property_magnitude', 'age',
            'other_payment_plans', 'housing', 'existing_credits', 'job',
            'num_dependents', 'own_telephone', 'foreign_worker', 'class'
        ]
        
        # Load data
        df = pd.read_csv(filepath, sep=' ', names=columns)
        
        # Convert target: 1 = good credit, 2 = bad credit
        # We'll convert to binary: 0 = good (no default), 1 = bad (default)
        df['default'] = (df['class'] == 2).astype(int)
        df = df.drop('class', axis=1)
        
        # Identify numerical and categorical features
        numerical_features = ['duration', 'credit_amount', 'installment_rate', 
                            'residence_since', 'age', 'existing_credits', 'num_dependents']
        
        categorical_features = [col for col in df.columns if col not in numerical_features + ['default']]
        
        # Encode categorical features
        le = LabelEncoder()
        for col in categorical_features:
            df[col] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def prepare_data(self, df, test_size=0.3, random_state=42):
        """
        Prepare data for modeling
        
        Args:
            df: DataFrame with features and 'default' target
            test_size: Proportion of test set
            random_state: Random seed
        """
        X = df.drop('default', axis=1)
        y = df['default']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale numerical features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Default rate in training: {self.y_train.mean():.2%}")
        print(f"Default rate in test: {self.y_test.mean():.2%}")
        
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*80)
        print("Training Logistic Regression...")
        print("="*80)
        
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # Predictions
        y_pred_proba = lr.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(self.y_test, y_pred_proba)
        ks_stat = self.calculate_ks_statistic(self.y_test, y_pred_proba)
        
        self.evaluation_results['Logistic Regression'] = {
            'auc': auc,
            'ks': ks_stat,
            'predictions': y_pred_proba
        }
        
        print(f"AUC-ROC: {auc:.4f}")
        print(f"KS-Statistic: {ks_stat:.4f}")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*80)
        print("Training Random Forest...")
        print("="*80)
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # Predictions
        y_pred_proba = rf.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(self.y_test, y_pred_proba)
        ks_stat = self.calculate_ks_statistic(self.y_test, y_pred_proba)
        
        self.evaluation_results['Random Forest'] = {
            'auc': auc,
            'ks': ks_stat,
            'predictions': y_pred_proba
        }
        
        print(f"AUC-ROC: {auc:.4f}")
        print(f"KS-Statistic: {ks_stat:.4f}")
        
    def train_gradient_boosting_monotonic(self):
        """Train Gradient Boosting with monotonic constraints using LightGBM"""
        print("\n" + "="*80)
        print("Training Gradient Boosting with Monotonic Constraints...")
        print("="*80)
        
        # Define monotonic constraints
        # Positive constraint (1): Higher feature value -> Higher default probability
        # Negative constraint (-1): Higher feature value -> Lower default probability
        # No constraint (0): No monotonicity enforced
        
        # Example constraints (adjust based on domain knowledge):
        # duration: longer loans might be riskier (+1)
        # credit_amount: higher amounts might be riskier (+1)
        # age: older applicants might be less risky (-1)
        
        monotone_constraints = {
            'duration': 1,
            'credit_amount': 1,
            'installment_rate': 1,
            'age': -1,
            'existing_credits': 0,
            'residence_since': 0,
            'num_dependents': 0
        }
        
        # Create constraint string for LightGBM
        constraint_list = []
        for feat in self.feature_names:
            if feat in monotone_constraints:
                constraint_list.append(monotone_constraints[feat])
            else:
                constraint_list.append(0)
        
        # Train LightGBM with monotonic constraints
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_test = lgb.Dataset(self.X_test, self.y_test, reference=lgb_train)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_data_in_leaf': 20,
            'monotone_constraints': constraint_list,
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True
        }
        
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=200,
            valid_sets=[lgb_test],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
        )
        
        self.models['Gradient Boosting (Monotonic)'] = gbm
        
        # Predictions
        y_pred_proba = gbm.predict(self.X_test, num_iteration=gbm.best_iteration)
        
        # Evaluate
        auc = roc_auc_score(self.y_test, y_pred_proba)
        ks_stat = self.calculate_ks_statistic(self.y_test, y_pred_proba)
        
        self.evaluation_results['Gradient Boosting (Monotonic)'] = {
            'auc': auc,
            'ks': ks_stat,
            'predictions': y_pred_proba
        }
        
        print(f"AUC-ROC: {auc:.4f}")
        print(f"KS-Statistic: {ks_stat:.4f}")
        
    def calculate_ks_statistic(self, y_true, y_pred_proba):
        """
        Calculate Kolmogorov-Smirnov statistic
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            KS statistic
        """
        # Separate scores for positive and negative classes
        scores_pos = y_pred_proba[y_true == 1]
        scores_neg = y_pred_proba[y_true == 0]
        
        # Calculate KS statistic
        ks_stat, _ = stats.ks_2samp(scores_pos, scores_neg)
        return ks_stat
    
    def plot_model_comparison(self):
        """Plot comparison of all models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # ROC Curves
        ax1 = axes[0]
        for model_name, results in self.evaluation_results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['predictions'])
            ax1.plot(fpr, tpr, label=f"{model_name} (AUC={results['auc']:.4f})")
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Performance Metrics Comparison
        ax2 = axes[1]
        models = list(self.evaluation_results.keys())
        auc_scores = [self.evaluation_results[m]['auc'] for m in models]
        ks_scores = [self.evaluation_results[m]['ks'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax2.bar(x - width/2, auc_scores, width, label='AUC-ROC', alpha=0.8)
        ax2.bar(x + width/2, ks_scores, width, label='KS-Statistic', alpha=0.8)
        
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=15, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig
    
    def initialize_shap_explainers(self):
        """Initialize SHAP explainers for all models"""
        print("\n" + "="*80)
        print("Initializing SHAP Explainers...")
        print("="*80)
        
        # Logistic Regression
        self.shap_explainers['Logistic Regression'] = shap.LinearExplainer(
            self.models['Logistic Regression'],
            self.X_train_scaled
        )
        
        # Random Forest
        self.shap_explainers['Random Forest'] = shap.TreeExplainer(
            self.models['Random Forest']
        )
        
        # Gradient Boosting
        self.shap_explainers['Gradient Boosting (Monotonic)'] = shap.TreeExplainer(
            self.models['Gradient Boosting (Monotonic)']
        )
        
        print("SHAP explainers initialized successfully!")
    
    def generate_shap_summary_plot(self, model_name='Gradient Boosting (Monotonic)'):
        """Generate SHAP summary plot for a specific model"""
        explainer = self.shap_explainers[model_name]
        
        if model_name == 'Logistic Regression':
            shap_values = explainer.shap_values(self.X_test_scaled)
            X_display = pd.DataFrame(self.X_test_scaled, columns=self.feature_names)
        else:
            shap_values = explainer.shap_values(self.X_test)
            X_display = self.X_test
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_display, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def generate_scorecard_with_reason_codes(self, applicant_data, model_name='Gradient Boosting (Monotonic)', top_n=5):
        """
        Generate scorecard with SHAP-based reason codes for an applicant
        
        Args:
            applicant_data: Dictionary or DataFrame row with applicant features
            model_name: Name of model to use
            top_n: Number of top reason codes to return
            
        Returns:
            Dictionary with risk score and reason codes
        """
        # Convert applicant data to DataFrame
        if isinstance(applicant_data, dict):
            applicant_df = pd.DataFrame([applicant_data])
        else:
            applicant_df = pd.DataFrame([applicant_data])
        
        # Ensure columns match training data
        applicant_df = applicant_df[self.feature_names]
        
        # Get model and explainer
        model = self.models[model_name]
        explainer = self.shap_explainers[model_name]
        
        # Make prediction
        if model_name == 'Logistic Regression':
            applicant_scaled = self.scaler.transform(applicant_df)
            risk_prob = model.predict_proba(applicant_scaled)[0, 1]
            shap_values = explainer.shap_values(applicant_scaled)[0]
        else:
            risk_prob = model.predict(applicant_df)[0]
            shap_values = explainer.shap_values(applicant_df)[0]
        
        # Convert risk probability to credit score (higher score = lower risk)
        credit_score = int((1 - risk_prob) * 850 + 300)  # Scale to 300-850 range
        
        # Get feature values
        feature_values = applicant_df.iloc[0].to_dict()
        
        # Create reason codes based on SHAP values
        # Positive SHAP values increase default risk (negative factors)
        # Negative SHAP values decrease default risk (positive factors)
        
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values,
            'feature_value': [feature_values[f] for f in self.feature_names]
        })
        
        # Top adverse factors (highest positive SHAP values)
        adverse_factors = shap_importance.nlargest(top_n, 'shap_value')
        
        # Top positive factors (most negative SHAP values)
        positive_factors = shap_importance.nsmallest(top_n, 'shap_value')
        
        # Create scorecard
        scorecard = {
            'credit_score': credit_score,
            'default_probability': float(risk_prob),
            'risk_category': self._get_risk_category(risk_prob),
            'adverse_factors': adverse_factors.to_dict('records'),
            'positive_factors': positive_factors.to_dict('records')
        }
        
        return scorecard
    
    def _get_risk_category(self, prob):
        """Categorize risk based on default probability"""
        if prob < 0.2:
            return 'Low Risk'
        elif prob < 0.4:
            return 'Medium Risk'
        elif prob < 0.6:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def plot_scorecard_explanation(self, scorecard):
        """Visualize scorecard explanation"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Adverse factors
        ax1 = axes[0]
        adverse_df = pd.DataFrame(scorecard['adverse_factors'])
        if len(adverse_df) > 0:
            adverse_df = adverse_df.sort_values('shap_value', ascending=True)
            colors = ['#ff6b6b' if x > 0 else '#51cf66' for x in adverse_df['shap_value']]
            ax1.barh(adverse_df['feature'], adverse_df['shap_value'], color=colors, alpha=0.7)
            ax1.set_xlabel('SHAP Value (Impact on Default Risk)', fontsize=11)
            ax1.set_title('Top Adverse Factors (Increasing Risk)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
        
        # Positive factors
        ax2 = axes[1]
        positive_df = pd.DataFrame(scorecard['positive_factors'])
        if len(positive_df) > 0:
            positive_df = positive_df.sort_values('shap_value', ascending=False)
            colors = ['#51cf66' if x < 0 else '#ff6b6b' for x in positive_df['shap_value']]
            ax2.barh(positive_df['feature'], positive_df['shap_value'], color=colors, alpha=0.7)
            ax2.set_xlabel('SHAP Value (Impact on Default Risk)', fontsize=11)
            ax2.set_title('Top Positive Factors (Decreasing Risk)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig

# ============================================================================
# Gradio Interface
# ============================================================================

def create_gradio_interface(scorer):
    """Create interactive Gradio interface"""
    
    def predict_credit_risk(duration, credit_amount, installment_rate, residence_since, 
                           age, existing_credits, num_dependents, checking_status,
                           credit_history, purpose, savings_status, employment,
                           personal_status, other_parties, property_magnitude,
                           other_payment_plans, housing, job, own_telephone, 
                           foreign_worker, model_choice):
        """Prediction function for Gradio interface"""
        
        # Create applicant data dictionary
        applicant_data = {
            'checking_status': int(checking_status),
            'duration': int(duration),
            'credit_history': int(credit_history),
            'purpose': int(purpose),
            'credit_amount': float(credit_amount),
            'savings_status': int(savings_status),
            'employment': int(employment),
            'installment_rate': int(installment_rate),
            'personal_status': int(personal_status),
            'other_parties': int(other_parties),
            'residence_since': int(residence_since),
            'property_magnitude': int(property_magnitude),
            'age': int(age),
            'other_payment_plans': int(other_payment_plans),
            'housing': int(housing),
            'existing_credits': int(existing_credits),
            'job': int(job),
            'num_dependents': int(num_dependents),
            'own_telephone': int(own_telephone),
            'foreign_worker': int(foreign_worker)
        }
        
        # Generate scorecard
        scorecard = scorer.generate_scorecard_with_reason_codes(
            applicant_data, 
            model_name=model_choice
        )
        
        # Create explanation plot
        fig = scorer.plot_scorecard_explanation(scorecard)
        
        # Format results
        result_text = f"""
        🎯 **Credit Score: {scorecard['credit_score']}**
        📊 **Default Probability: {scorecard['default_probability']:.2%}**
        ⚠️ **Risk Category: {scorecard['risk_category']}**
        
        ### Top Adverse Factors (Increasing Risk):
        """
        
        for i, factor in enumerate(scorecard['adverse_factors'], 1):
            result_text += f"\n{i}. **{factor['feature']}** (value: {factor['feature_value']:.2f}, impact: {factor['shap_value']:.4f})"
        
        result_text += "\n\n### Top Positive Factors (Decreasing Risk):"
        
        for i, factor in enumerate(scorecard['positive_factors'], 1):
            result_text += f"\n{i}. **{factor['feature']}** (value: {factor['feature_value']:.2f}, impact: {factor['shap_value']:.4f})"
        
        return result_text, fig
    
    # Create Gradio interface
    with gr.Blocks(title="Credit Risk Scoring System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏦 Credit Risk Scoring System
        ## AI-Powered Credit Assessment with Explainable AI
        
        This system uses advanced machine learning models with SHAP (SHapley Additive exPlanations) 
        to provide transparent credit risk assessments and reason codes for each decision.
        """)
        
        with gr.Tab("📋 Credit Assessment"):
            gr.Markdown("### Enter Applicant Information")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Financial Information")
                    duration = gr.Number(label="Loan Duration (months)", value=24)
                    credit_amount = gr.Number(label="Credit Amount", value=5000)
                    installment_rate = gr.Slider(1, 4, value=2, step=1, label="Installment Rate (% of disposable income)")
                    existing_credits = gr.Slider(1, 4, value=1, step=1, label="Number of Existing Credits")
                    
                    gr.Markdown("#### Status Codes")
                    checking_status = gr.Slider(0, 3, value=1, step=1, label="Checking Account Status (0-3)")
                    savings_status = gr.Slider(0, 4, value=1, step=1, label="Savings Account Status (0-4)")
                    credit_history = gr.Slider(0, 4, value=2, step=1, label="Credit History (0-4)")
                    purpose = gr.Slider(0, 9, value=3, step=1, label="Purpose of Loan (0-9)")
                
                with gr.Column():
                    gr.Markdown("#### Personal Information")
                    age = gr.Number(label="Age", value=35)
                    residence_since = gr.Slider(1, 4, value=2, step=1, label="Years at Current Residence")
                    num_dependents = gr.Slider(1, 2, value=1, step=1, label="Number of Dependents")
                    
                    gr.Markdown("#### Employment & Housing")
                    employment = gr.Slider(0, 4, value=2, step=1, label="Employment Status (0-4)")
                    job = gr.Slider(0, 3, value=2, step=1, label="Job Category (0-3)")
                    housing = gr.Slider(0, 2, value=1, step=1, label="Housing Status (0-2)")
                    property_magnitude = gr.Slider(0, 3, value=2, step=1, label="Property (0-3)")
                    
                    gr.Markdown("#### Other Factors")
                    personal_status = gr.Slider(0, 3, value=2, step=1, label="Personal Status (0-3)")
                    other_parties = gr.Slider(0, 2, value=0, step=1, label="Other Parties (0-2)")
                    other_payment_plans = gr.Slider(0, 2, value=0, step=1, label="Other Payment Plans (0-2)")
                    own_telephone = gr.Slider(0, 1, value=1, step=1, label="Own Telephone (0=No, 1=Yes)")
                    foreign_worker = gr.Slider(0, 1, value=1, step=1, label="Foreign Worker (0=No, 1=Yes)")
            
            model_choice = gr.Radio(
                choices=['Gradient Boosting (Monotonic)', 'Logistic Regression', 'Random Forest'],
                value='Gradient Boosting (Monotonic)',
                label="Select Model"
            )
            
            predict_btn = gr.Button("🔍 Assess Credit Risk", variant="primary", size="lg")
            
            with gr.Row():
                result_text = gr.Markdown(label="Assessment Results")
            
            with gr.Row():
                explanation_plot = gr.Plot(label="SHAP Explanation")
            
            predict_btn.click(
                fn=predict_credit_risk,
                inputs=[duration, credit_amount, installment_rate, residence_since, 
                       age, existing_credits, num_dependents, checking_status,
                       credit_history, purpose, savings_status, employment,
                       personal_status, other_parties, property_magnitude,
                       other_payment_plans, housing, job, own_telephone, 
                       foreign_worker, model_choice],
                outputs=[result_text, explanation_plot]
            )
        
        with gr.Tab("📊 Model Performance"):
            gr.Markdown("### Model Comparison and Performance Metrics")
            
            comparison_plot = gr.Plot(value=scorer.plot_model_comparison())
            
            gr.Markdown("### Performance Summary")
            
            perf_data = []
            for model_name, results in scorer.evaluation_results.items():
                perf_data.append([model_name, f"{results['auc']:.4f}", f"{results['ks']:.4f}"])
            
            perf_df = pd.DataFrame(perf_data, columns=['Model', 'AUC-ROC', 'KS-Statistic'])
            gr.Dataframe(value=perf_df, label="Model Performance Metrics")
        
        with gr.Tab("🔍 SHAP Analysis"):
            gr.Markdown("### SHAP Feature Importance Analysis")
            
            shap_model_choice = gr.Radio(
                choices=['Gradient Boosting (Monotonic)', 'Logistic Regression', 'Random Forest'],
                value='Gradient Boosting (Monotonic)',
                label="Select Model for SHAP Analysis"
            )
            
            shap_btn = gr.Button("Generate SHAP Summary Plot", variant="secondary")
            shap_plot = gr.Plot(label="SHAP Summary")
            
            shap_btn.click(
                fn=lambda model: scorer.generate_shap_summary_plot(model),
                inputs=[shap_model_choice],
                outputs=[shap_plot]
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This System
            
            This Credit Risk Scoring System implements state-of-the-art machine learning techniques for credit assessment:
            
            ### Models Implemented:
            1. **Gradient Boosting with Monotonic Constraints**: Ensures predictions follow logical relationships (e.g., higher age reduces risk)
            2. **Logistic Regression**: Traditional baseline model for interpretability
            3. **Random Forest**: Ensemble method for robust predictions
            
            ### Key Features:
            - **SHAP Explanations**: Understand exactly why each decision was made
            - **Monotonic Constraints**: Ensure logical relationships between features and predictions
            - **Reason Codes**: Regulatory-compliant explanations for adverse actions
            - **Multiple Metrics**: AUC-ROC and KS-statistic for comprehensive evaluation
            
            ### Dataset:
            - **German Credit Data** from UCI Machine Learning Repository
            - 1,000 loan applications with 20 features
            - Binary classification: Good credit vs. Bad credit (default)
            
            ### Evaluation Metrics:
            - **AUC-ROC**: Measures model's ability to discriminate between classes
            - **KS-Statistic**: Measures separation between good and bad credit distributions
            
            ### Usage Guidelines:
            1. Enter applicant information in the "Credit Assessment" tab
            2. Select your preferred model
            3. Click "Assess Credit Risk" to get score and explanation
            4. Review adverse and positive factors to understand the decision
            
            ### Regulatory Compliance:
            This system provides transparent reason codes that can be used for adverse action notices
            as required by fair lending regulations.
            """)
    
    return demo

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    
    print("""
    ================================================================================
                        Credit Risk Scoring System
    ================================================================================
    """)
    
    # Initialize scorer
    scorer = CreditRiskScorer()
    
    # Load data
    print("\nStep 1: Loading German Credit Data...")
    print("-" * 80)
    print("Please ensure you have downloaded the dataset from:")
    print("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
    print("\nSave it as 'german.data' in the same directory as this script.")
    
    filepath = input("\nEnter the path to german.data file (or press Enter for default 'german.data'): ").strip()
    if not filepath:
        filepath = 'german.data'
    
    try:
        df = scorer.load_german_credit_data(filepath)
        print(f"\n✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"✓ Default rate: {df['default'].mean():.2%}")
    except FileNotFoundError:
        print(f"\n✗ Error: File '{filepath}' not found!")
        print("\nPlease download the dataset from:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
        return
    
    # Prepare data
    print("\n\nStep 2: Preparing Data...")
    print("-" * 80)
    scorer.prepare_data(df)
    
    # Train models
    print("\n\nStep 3: Training Models...")
    print("-" * 80)
    
    scorer.train_logistic_regression()
    scorer.train_random_forest()
    scorer.train_gradient_boosting_monotonic()
    
    # Plot comparison
    print("\n\nStep 4: Generating Model Comparison...")
    print("-" * 80)
    fig = scorer.plot_model_comparison()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Model comparison plot saved as 'model_comparison.png'")
    
    # Initialize SHAP
    print("\n\nStep 5: Initializing SHAP Explainers...")
    print("-" * 80)
    scorer.initialize_shap_explainers()
    
    # Generate SHAP summary
    print("\n\nStep 6: Generating SHAP Summary Plots...")
    print("-" * 80)
    for model_name in ['Gradient Boosting (Monotonic)', 'Random Forest', 'Logistic Regression']:
        fig = scorer.generate_shap_summary_plot(model_name)
        filename = f"shap_summary_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP summary plot saved as '{filename}'")
        plt.close()
    
    # Example scorecard generation
    print("\n\nStep 7: Generating Example Scorecard...")
    print("-" * 80)
    
    # Use first test sample as example
    example_applicant = scorer.X_test.iloc[0].to_dict()
    scorecard = scorer.generate_scorecard_with_reason_codes(
        example_applicant, 
        model_name='Gradient Boosting (Monotonic)'
    )
    
    print(f"\n{'='*80}")
    print("EXAMPLE CREDIT SCORECARD")
    print(f"{'='*80}")
    print(f"\n🎯 Credit Score: {scorecard['credit_score']}")
    print(f"📊 Default Probability: {scorecard['default_probability']:.2%}")
    print(f"⚠️  Risk Category: {scorecard['risk_category']}")
    
    print("\n--- TOP ADVERSE FACTORS (Increasing Risk) ---")
    for i, factor in enumerate(scorecard['adverse_factors'], 1):
        print(f"{i}. {factor['feature']}: value={factor['feature_value']:.2f}, impact={factor['shap_value']:.4f}")
    
    print("\n--- TOP POSITIVE FACTORS (Decreasing Risk) ---")
    for i, factor in enumerate(scorecard['positive_factors'], 1):
        print(f"{i}. {factor['feature']}: value={factor['feature_value']:.2f}, impact={factor['shap_value']:.4f}")
    
    # Generate scorecard plot
    fig = scorer.plot_scorecard_explanation(scorecard)
    plt.savefig('example_scorecard.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Example scorecard plot saved as 'example_scorecard.png'")
    plt.close()
    
    # Print final summary
    print(f"\n\n{'='*80}")
    print("MODEL PERFORMANCE SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<35} {'AUC-ROC':<12} {'KS-Statistic':<15}")
    print("-" * 80)
    for model_name, results in scorer.evaluation_results.items():
        print(f"{model_name:<35} {results['auc']:<12.4f} {results['ks']:<15.4f}")
    
    # Launch Gradio interface
    print(f"\n\n{'='*80}")
    print("LAUNCHING INTERACTIVE INTERFACE")
    print(f"{'='*80}\n")
    print("Starting Gradio interface...")
    print("The interface will open in your default browser.")
    print("You can also access it at the local URL that will be displayed.\n")
    
    demo = create_gradio_interface(scorer)
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()