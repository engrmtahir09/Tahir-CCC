from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import pickle
import os
from typing import Dict, List, Optional
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="Concrete Properties Prediction & Mix Design API")

# Serve static files (images)
app.mount("/static", StaticFiles(directory="."), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and data
try:
    with open('concrete_models.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    bpnn_models = model_data['bpnn_models']
    bpnn_x_scalers = model_data['bpnn_x_scalers']
    bpnn_y_scalers = model_data['bpnn_y_scalers']
    input_vars = model_data['input_vars']
    output_vars = model_data['output_vars']
    material_costs = model_data['material_costs']
    transport_distance = model_data['transport_distance']
    material_densities = model_data['material_densities']
    co2_factors = model_data['co2_factors']
    
    print("Models loaded successfully!")
except FileNotFoundError:
    raise Exception("Model file not found. Please run the training notebook first and export models.")

# Constants
transport_cost_per_km_kg = 0.000112
compaction_cost_per_minute = 1.0377
compaction_co2_per_minute = 4.177
transport_co2_per_km_kg = 0.000078

# Pydantic models
class ConcreteInput(BaseModel):
    cement: float = Field(..., ge=190, le=612, description="Cement content (kg/m³)")
    fine_aggregate: float = Field(..., ge=480, le=945, description="Fine aggregate content (kg/m³)")
    coarse_aggregate: float = Field(..., ge=806, le=1175, description="Coarse aggregate content (kg/m³)")
    water: float = Field(..., ge=100, le=266, description="Water content (kg/m³)")
    casting_pressure: float = Field(..., ge=0, le=15, description="Casting pressure (MPa)")

class ConcretePrediction(BaseModel):
    density: float
    strength: float
    elastic_modulus: float
    peak_strain: float

class CostCO2Factors(BaseModel):
    cement_cost: Optional[float] = None
    fine_agg_cost: Optional[float] = None
    coarse_agg_cost: Optional[float] = None
    water_cost: Optional[float] = None
    cement_co2: Optional[float] = None
    fine_agg_co2: Optional[float] = None
    coarse_agg_co2: Optional[float] = None
    water_co2: Optional[float] = None
    cement_transport_distance: Optional[float] = None
    fine_agg_transport_distance: Optional[float] = None
    coarse_agg_transport_distance: Optional[float] = None
    water_transport_distance: Optional[float] = None

class MixDesignInput(BaseModel):
    target_strength: float = Field(..., ge=20, le=65, description="Target compressive strength (MPa)")
    cost_weight: Optional[float] = Field(0.334, ge=0, le=1, description="Weight for cost optimization")
    co2_weight: Optional[float] = Field(0.333, ge=0, le=1, description="Weight for CO2 optimization")
    density_weight: Optional[float] = Field(0.333, ge=0, le=1, description="Weight for density optimization")
    cost_co2_factors: Optional[CostCO2Factors] = Field(None, description="Optional custom cost and CO2 factors")

class MixDesignResult(BaseModel):
    rank: int
    cement: float
    fine_aggregate: float
    coarse_aggregate: float
    water: float
    casting_pressure: float
    cost: float
    co2_emissions: float
    predicted_density: float
    predicted_strength: float
    predicted_elastic_modulus: float
    predicted_peak_strain: float
    topsis_score: float

# Utility functions
def calculate_cost(mix, casting_pressure, custom_costs=None, custom_transport=None):
    # Use custom costs if provided, otherwise use defaults
    costs = {
        'CC': custom_costs.get('cement_cost', material_costs['CC']) if custom_costs else material_costs['CC'],
        'FA': custom_costs.get('fine_agg_cost', material_costs['FA']) if custom_costs else material_costs['FA'],
        'CA': custom_costs.get('coarse_agg_cost', material_costs['CA']) if custom_costs else material_costs['CA'],
        'WC': custom_costs.get('water_cost', material_costs['WC']) if custom_costs else material_costs['WC']
    }
    
    transport = {
        'CC': custom_transport.get('cement_transport_distance', transport_distance['CC']) if custom_transport else transport_distance['CC'],
        'FA': custom_transport.get('fine_agg_transport_distance', transport_distance['FA']) if custom_transport else transport_distance['FA'],
        'CA': custom_transport.get('coarse_agg_transport_distance', transport_distance['CA']) if custom_transport else transport_distance['CA'],
        'WC': custom_transport.get('water_transport_distance', transport_distance['WC']) if custom_transport else transport_distance['WC']
    }
    
    material_cost = (mix[0] * costs['CC'] +
                    mix[1] * costs['FA'] +
                    mix[2] * costs['CA'] +
                    mix[3] * costs['WC'])
    
    transport_cost = (mix[0] * transport['CC'] +  
                    mix[1] * transport['FA'] + 
                    mix[2] * transport['CA'] +  
                    mix[3] * transport['WC']) * transport_cost_per_km_kg         

    compaction_cost = 2 * compaction_cost_per_minute if casting_pressure > 0 else 0.5 * compaction_cost_per_minute
    return material_cost + transport_cost + compaction_cost

def calculate_co2(mix, casting_pressure, custom_co2=None, custom_transport=None):
    # Use custom CO2 factors if provided, otherwise use defaults
    co2_vals = {
        'CC': custom_co2.get('cement_co2', co2_factors['CC']) if custom_co2 else co2_factors['CC'],
        'FA': custom_co2.get('fine_agg_co2', co2_factors['FA']) if custom_co2 else co2_factors['FA'],
        'CA': custom_co2.get('coarse_agg_co2', co2_factors['CA']) if custom_co2 else co2_factors['CA'],
        'WC': custom_co2.get('water_co2', co2_factors['WC']) if custom_co2 else co2_factors['WC']
    }
    
    transport = {
        'CC': custom_transport.get('cement_transport_distance', transport_distance['CC']) if custom_transport else transport_distance['CC'],
        'FA': custom_transport.get('fine_agg_transport_distance', transport_distance['FA']) if custom_transport else transport_distance['FA'],
        'CA': custom_transport.get('coarse_agg_transport_distance', transport_distance['CA']) if custom_transport else transport_distance['CA'],
        'WC': custom_transport.get('water_transport_distance', transport_distance['WC']) if custom_transport else transport_distance['WC']
    }
    
    material_co2 = (mix[0] * co2_vals['CC'] +
                   mix[1] * co2_vals['FA'] +
                   mix[2] * co2_vals['CA'] +
                   mix[3] * co2_vals['WC'])

    transport_co2 = (mix[0] * transport['CC'] +  
                    mix[1] * transport['FA'] + 
                    mix[2] * transport['CA'] +  
                    mix[3] * transport['WC']) * transport_co2_per_km_kg         

    compaction_co2 = 2 * compaction_co2_per_minute if casting_pressure > 0 else 0.5 * compaction_co2_per_minute
    return material_co2 + transport_co2 + compaction_co2

def predict_properties(input_data):
    """Predict concrete properties using trained models"""
    full_input = np.array([
        input_data.cement,
        input_data.fine_aggregate,
        input_data.coarse_aggregate,
        input_data.water,
        input_data.casting_pressure
    ]).reshape(1, -1)
    
    predictions = {}
    for target in output_vars:
        model = bpnn_models[target]
        x_scaler = bpnn_x_scalers[target]
        y_scaler = bpnn_y_scalers[target]
        
        X_scaled = x_scaler.transform(full_input)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
        predictions[target] = y_pred
    
    return predictions

# Multi-objective optimization problem
class ConcreteMixProblem(Problem):
    def __init__(self, target_strength, custom_costs=None, custom_co2=None, custom_transport=None):
        n_var = 5  # CC, FA, CA, WC, P
        xl = [190, 480, 806, 100, 0]  # Lower bounds
        xu = [612, 945, 1175, 266, 15]  # Upper bounds
        n_obj = 3  # cost, CO2, density
        n_constr = 5  # volume, w/c ratio, FA ratio, strength bounds
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.target_strength = target_strength
        self.custom_costs = custom_costs
        self.custom_co2 = custom_co2
        self.custom_transport = custom_transport
        
    def _evaluate(self, X, out, *args, **kwargs):
        n_samples = X.shape[0]
        f = np.zeros((n_samples, self.n_obj))
        g = np.zeros((n_samples, self.n_constr))
        
        for i in range(n_samples):
            mix = X[i]
            cc, fa, ca, wc, P = mix
            
            full_input = np.array([cc, fa, ca, wc, P])
            
            # Predict properties
            preds = {}
            for target, model in bpnn_models.items():
                scaled_input = bpnn_x_scalers[target].transform(full_input.reshape(1, -1))
                scaled_pred = model.predict(scaled_input)
                preds[target] = bpnn_y_scalers[target].inverse_transform(scaled_pred.reshape(-1, 1))[0, 0]
            
            # Objectives - use custom factors if provided
            f[i, 0] = calculate_cost(full_input, P, self.custom_costs, self.custom_transport)  # Cost
            f[i, 1] = calculate_co2(full_input, P, self.custom_co2, self.custom_transport)  # CO2
            f[i, 2] = -preds['pc']  # Negative density (to maximize)
            
            # Constraints
            volumes = np.array([
                cc / material_densities['CC'],
                fa / material_densities['FA'],
                ca / material_densities['CA'],
                wc / material_densities['WC']
            ])
            total_volume = np.sum(volumes)
            g[i, 0] = abs(total_volume - 1) - 0.01  # Volume constraint
            
            wc_ratio = wc / cc
            g[i, 1] = max(0.35 - wc_ratio, wc_ratio - 0.65, 0)  # W/C ratio
            
            fa_ratio = fa / (fa + ca)
            g[i, 2] = max(0.3 - fa_ratio, fa_ratio - 0.45, 0)  # FA ratio
            
            # Strength constraints
            g[i, 3] = self.target_strength - preds['fc']  # Lower bound
            g[i, 4] = preds['fc'] - (self.target_strength + 2.5)  # Upper bound
            
        out["F"] = f
        out["G"] = g

def get_predictions_from_mix(x):
    """Get predictions for a mix design"""
    full_input = x
    preds = {}
    for target, model in bpnn_models.items():
        scaled_input = bpnn_x_scalers[target].transform(full_input.reshape(1, -1))
        scaled_pred = model.predict(scaled_input)
        preds[target] = bpnn_y_scalers[target].inverse_transform(scaled_pred.reshape(-1, 1))[0, 0]
    return preds

def optimize_mix_design(target_strength, weights, custom_costs=None, custom_co2=None, custom_transport=None):
    """Optimize concrete mix design"""
    problem = ConcreteMixProblem(target_strength, custom_costs, custom_co2, custom_transport)
    
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=20)
    
    algorithm = NSGA3(
        pop_size=250,  # Reduced for faster API response
        ref_dirs=ref_dirs,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=10),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", 150)  # Reduced for faster response
    
    res = minimize(problem, algorithm, termination, seed=42, verbose=False)
    
    # Apply TOPSIS selection
    F = res.F
    G = res.G
    X = res.X
    
    # Get feasible solutions
    feasible_mask = np.all(G <= 0, axis=1)
    if not np.any(feasible_mask):
        # If no feasible solutions, return best infeasible
        cv = np.sum(np.maximum(G, 0), axis=1)
        best_idx = np.argmin(cv)
        return [X[best_idx]], [F[best_idx]], [0.0]
    
    feasible_F = F[feasible_mask]
    feasible_X = X[feasible_mask]
    
    # Normalize and apply TOPSIS
    weight_array = np.array([weights['cost'], weights['co2'], weights['density']])
    weight_array = weight_array / np.sum(weight_array)
    
    norm_F = (feasible_F - feasible_F.min(axis=0)) / (feasible_F.max(axis=0) - feasible_F.min(axis=0) + 1e-10)
    weighted_norm_F = norm_F * weight_array
    
    ideal = np.array([np.min(weighted_norm_F[:, 0]), np.min(weighted_norm_F[:, 1]), np.max(weighted_norm_F[:, 2])])
    neg_ideal = np.array([np.max(weighted_norm_F[:, 0]), np.max(weighted_norm_F[:, 1]), np.min(weighted_norm_F[:, 2])])
    
    d_pos = np.sqrt(np.sum((weighted_norm_F - ideal)**2, axis=1))
    d_neg = np.sqrt(np.sum((weighted_norm_F - neg_ideal)**2, axis=1))
    
    topsis_score = d_neg / (d_pos + d_neg + 1e-10)
    
    # Get top 10 solutions
    top10_idx = np.argsort(topsis_score)[-10:][::-1]
    
    return feasible_X[top10_idx], feasible_F[top10_idx], topsis_score[top10_idx]

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Concrete Properties Prediction & Mix Design System</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary-color: #6366f1;
                --primary-dark: #4f46e5;
                --primary-light: #8b5cf6;
                --secondary-color: #64748b;
                --success-color: #10b981;
                --success-light: #34d399;
                --warning-color: #f59e0b;
                --warning-light: #fbbf24;
                --danger-color: #ef4444;
                --danger-light: #f87171;
                --info-color: #06b6d4;
                --info-light: #22d3ee;
                --light-bg: #f8fafc;
                --dark-bg: #0f172a;
                --card-bg: #ffffff;
                --border-color: #e2e8f0;
                --text-primary: #1e293b;
                --text-secondary: #64748b;
                --shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                --gradient-4: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                --gradient-5: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Times New Roman', Times, serif;
                background: var(--gradient-1);
                background-size: 400% 400%;
                animation: gradientShift 8s ease infinite;
                min-height: 100vh;
                padding: 2rem 1rem;
                color: var(--text-primary);
                position: relative;
                overflow-x: hidden;
            }

            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: 
                    radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 40% 80%, rgba(120, 219, 255, 0.3) 0%, transparent 50%);
                pointer-events: none;
                z-index: -1;
                animation: floatingOrbs 12s ease-in-out infinite;
            }

            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            @keyframes floatingOrbs {
                0%, 100% { transform: translate(0, 0) scale(1); }
                33% { transform: translate(30px, -30px) scale(1.1); }
                66% { transform: translate(-20px, 20px) scale(0.9); }
            }

            .main-container {
                max-width: 1400px;
                margin: 0 auto;
                background: var(--card-bg);
                border-radius: 20px;
                box-shadow: var(--shadow-lg);
                overflow: hidden;
                position: relative;
                animation: slideInUp 1s ease-out;
            }

            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(60px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .main-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--gradient-2);
                background-size: 200% 200%;
                animation: gradientShift 3s ease infinite;
            }

            .header {
                background: var(--gradient-3);
                background-size: 400% 400%;
                animation: gradientShift 6s ease infinite;
                color: white;
                padding: 3rem 2rem;
                text-align: center;
                position: relative;
                overflow: hidden;
            }

            .header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Cpath d='m36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
                animation: float 20s ease-in-out infinite;
                z-index: 0;
            }

            .header::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: 
                    radial-gradient(circle at 30% 40%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 70% 70%, rgba(255, 255, 255, 0.08) 0%, transparent 50%);
                animation: pulse 4s ease-in-out infinite;
                z-index: 1;
            }

            .header-content {
                position: relative;
                z-index: 1;
            }

            .header h1 {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                text-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                animation: titleGlow 3s ease-in-out infinite alternate;
                position: relative;
                z-index: 2;
            }

            @keyframes titleGlow {
                from {
                    text-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), 0 0 20px rgba(255, 255, 255, 0.1);
                }
                to {
                    text-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), 0 0 30px rgba(255, 255, 255, 0.3);
                }
            }

            .header p {
                font-size: 1.2rem;
                opacity: 0.9;
                max-width: 600px;
                margin: 0 auto;
                line-height: 1.6;
                position: relative;
                z-index: 2;
                animation: fadeInUp 1s ease-out 0.5s both;
            }

            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 0.9;
                    transform: translateY(0);
                }
            }

            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                50% { transform: translateY(-20px) rotate(180deg); }
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }

            @keyframes bounce {
                0%, 20%, 53%, 80%, 100% { transform: translate3d(0, 0, 0); }
                40%, 43% { transform: translate3d(0, -15px, 0); }
                70% { transform: translate3d(0, -7px, 0); }
                90% { transform: translate3d(0, -2px, 0); }
            }

            .content {
                padding: 3rem 2rem;
            }

            /* Tab Navigation Styles */
            .tab-navigation {
                display: flex;
                background: #f8fafc;
                border-radius: 12px;
                padding: 8px;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow-x: auto;
            }

            .tab-button {
                flex: 1;
                min-width: 200px;
                padding: 1rem 1.5rem;
                border: none;
                background: transparent;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
                font-size: 0.95rem;
                color: var(--text-secondary);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                white-space: nowrap;
                position: relative;
                overflow: hidden;
            }

            .tab-button::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
                transition: left 0.5s;
            }

            .tab-button:hover::before {
                left: 100%;
            }

            .tab-button i {
                margin-right: 0.5rem;
                font-size: 1.1rem;
            }

            .tab-button:hover {
                background: rgba(99, 102, 241, 0.1);
                color: var(--primary-color);
                transform: translateY(-2px);
            }

            .tab-button.active {
                background: var(--primary-color);
                color: white;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
                transform: translateY(-1px);
            }

            .tab-button.active:hover {
                background: var(--primary-dark);
                transform: translateY(-2px);
            }

            /* Tab Content Styles */
            .tab-content {
                display: none;
                animation: fadeInUp 0.6s ease-out;
            }

            .tab-content.active {
                display: block;
            }

            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 2rem;
                margin-bottom: 2rem;
            }

            @media (max-width: 1024px) {
                .grid {
                    grid-template-columns: 1fr;
                    gap: 2rem;
                }
                
                .tab-button {
                    min-width: 160px;
                    padding: 0.8rem 1rem;
                    font-size: 0.9rem;
                }
            }

            @media (max-width: 768px) {
                .grid {
                    grid-template-columns: 1fr;
                    gap: 2rem;
                }
                
                .tab-navigation {
                    flex-direction: column;
                    gap: 0.5rem;
                }
                
                .tab-button {
                    min-width: auto;
                    flex: none;
                }
                
                .about-content {
                    grid-template-columns: 1fr !important;
                    gap: 1.5rem !important;
                }
                
                .about-images {
                    grid-template-columns: 1fr !important;
                }
                
                .ccc-content {
                    grid-template-columns: 1fr !important;
                }
                
                .header-content div[style*="grid-template-columns: 1fr 1fr"] {
                    grid-template-columns: 1fr !important;
                    gap: 1rem !important;
                    max-width: 400px !important;
                }
                
                .header-content img {
                    height: 220px !important;
                }
            }

            .section {
                background: var(--card-bg);
                border-radius: 16px;
                padding: 2rem;
                box-shadow: var(--shadow);
                border: 1px solid var(--border-color);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
                animation: fadeInScale 0.8s ease-out;
            }

            @keyframes fadeInScale {
                from {
                    opacity: 0;
                    transform: scale(0.95);
                }
                to {
                    opacity: 1;
                    transform: scale(1);
                }
            }

            .section::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
                transition: left 0.8s;
            }

            .section:hover {
                transform: translateY(-8px) scale(1.02);
                box-shadow: 0 25px 50px -5px rgba(0, 0, 0, 0.15), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                border-color: var(--primary-color);
            }

            .section:hover::before {
                left: 100%;
            }

            .section-header {
                display: flex;
                align-items: center;
                margin-bottom: 1.0rem;
                padding-bottom: 1rem;
                border-bottom: 2px solid var(--light-bg);
            }

            .section-icon {
                width: 50px;
                height: 50px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 1rem;
                font-size: 1.5rem;
            }

            .prediction-icon {
                background: var(--gradient-4);
                color: white;
                animation: iconPulse 2s ease-in-out infinite;
            }

            .design-icon {
                background: var(--gradient-5);
                color: white;
                animation: iconPulse 2s ease-in-out infinite 0.5s;
            }

            @keyframes iconPulse {
                0%, 100% {
                    transform: scale(1);
                    box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.4);
                }
                50% {
                    transform: scale(1.05);
                    box-shadow: 0 0 0 10px rgba(255, 255, 255, 0);
                }
            }

            .section h2 {
                font-size: 1.5rem;
                font-weight: 600;
                color: var(--text-primary);
                margin: 0;
            }

            .section p {
                color: var(--text-secondary);
                margin-bottom: 1.5rem;
                line-height: 1.6;
            }

            .form-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.8rem;
                margin-bottom: 0.5rem;
            }

            @media (max-width: 640px) {
                .form-grid {
                    grid-template-columns: 1fr;
                }
            }

            .form-group {
                margin-bottom: 0.5rem;
            }

            .form-group label {
                display: flex;
                align-items: center;
                margin-bottom: 0.5rem;
                font-weight: 500;
                color: var(--text-primary);
                font-size: 0.9rem;
            }

            .form-group label i {
                margin-right: 0.5rem;
                color: var(--primary-color);
                width: 16px;
            }

            .form-input {
                width: 100%;
                padding: 0.7rem 0.7rem;
                border: 2px solid var(--border-color);
                border-radius: 8px;
                font-size: 0.9rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                background: var(--card-bg);
                position: relative;
            }

            .form-input:focus {
                outline: none;
                border-color: var(--primary-color);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
                transform: translateY(-2px);
            }

            .form-input:hover:not(:focus) {
                border-color: var(--primary-light);
                transform: translateY(-1px);
            }

            .form-input:disabled {
                background-color: #f1f5f9;
                color: #64748b;
                cursor: not-allowed;
                opacity: 0.7;
            }

            .form-input:disabled:hover {
                border-color: var(--border-color);
                transform: none;
            }

            .btn {
                background: var(--gradient-3);
                background-size: 200% 200%;
                color: white;
                border: none;
                padding: 0.875rem 2rem;
                border-radius: 8px;
                font-size: 0.9rem;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                text-decoration: none;
                margin-right: 1rem;
                margin-bottom: 0.8rem;
                position: relative;
                overflow: hidden;
            }

            .btn::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: left 0.5s;
            }

            .btn:hover::before {
                left: 100%;
            }

            .btn i {
                margin-right: 0.5rem;
                animation: bounce 2s infinite;
            }

            .btn:hover {
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 15px 30px rgba(79, 70, 229, 0.4);
                background-position: right center;
            }

            .btn:active {
                transform: translateY(-1px) scale(1.02);
            }

            .btn-success {
                background: var(--gradient-4);
                background-size: 200% 200%;
            }

            .btn-success:hover {
                box-shadow: 0 15px 30px rgba(16, 185, 129, 0.4);
                background-position: right center;
            }

            .results {
                background: linear-gradient(135deg, var(--light-bg) 0%, #ffffff 100%);
                border-radius: 12px;
                padding: 1.0rem;
                margin-top: 1.0rem;
                border: 1px solid var(--border-color);
                display: none;
                animation: slideInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }

            .results::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: var(--gradient-2);
                background-size: 200% 200%;
                animation: gradientShift 2s ease infinite;
            }

            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px) scale(0.95);
                }
                to {
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }
            }

            .results h3 {
                color: var(--text-primary);
                margin-bottom: 0.8rem;
                font-weight: 600;
                display: flex;
                align-items: center;
            }

            .results h3 i {
                margin-right: 0.5rem;
                color: var(--success-color);
            }

            .result-item {
                background: white;
                padding: 1rem;
                margin: 0.7rem 0;
                border-radius: 8px;
                border-left: 4px solid var(--primary-color);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: all 0.3s ease;
                animation: fadeInLeft 0.5s ease forwards;
                opacity: 0;
                transform: translateX(-20px);
            }

            @keyframes fadeInLeft {
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }

            .result-item:nth-child(1) { animation-delay: 0.1s; }
            .result-item:nth-child(2) { animation-delay: 0.2s; }
            .result-item:nth-child(3) { animation-delay: 0.3s; }
            .result-item:nth-child(4) { animation-delay: 0.4s; }
            .result-item:nth-child(5) { animation-delay: 0.5s; }
            .result-item:nth-child(6) { animation-delay: 0.6s; }

            .result-item:hover {
                transform: translateX(5px) scale(1.02);
                box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
                border-left-width: 6px;
            }

            .result-label {
                font-weight: 500;
                color: var(--text-primary);
                display: flex;
                align-items: center;
            }

            .result-label i {
                margin-right: 0.5rem;
                color: var(--primary-color);
                width: 20px;
            }

            .result-value {
                font-weight: 600;
                color: var(--success-color);
                font-size: 1.1rem;
            }

            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 0.8rem;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }

            .data-table th {
                background: var(--primary-color);
                color: white;
                padding: 1rem 0.75rem;
                text-align: left;
                font-weight: 500;
                font-size: 0.875rem;
            }

            .data-table td {
                padding: 0.75rem;
                border-bottom: 1px solid var(--border-color);
                font-size: 0.875rem;
            }

            .data-table tr:hover {
                background: var(--light-bg);
            }

            .loading {
                display: inline-flex;
                align-items: center;
                color: var(--text-secondary);
            }

            .loading i {
                animation: spin 1s linear infinite;
                margin-right: 0.5rem;
                color: var(--primary-color);
            }

            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            .loading-dots {
                display: inline-flex;
                margin-left: 0.5rem;
            }

            .loading-dots span {
                width: 4px;
                height: 4px;
                border-radius: 50%;
                background: var(--primary-color);
                margin: 0 2px;
                animation: loadingDots 1.4s ease-in-out infinite both;
            }

            .loading-dots span:nth-child(1) { animation-delay: -0.32s; }
            .loading-dots span:nth-child(2) { animation-delay: -0.16s; }
            .loading-dots span:nth-child(3) { animation-delay: 0s; }

            @keyframes loadingDots {
                0%, 80%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                40% {
                    transform: scale(1.2);
                    opacity: 1;
                }
            }

            .alert {
                padding: 1rem;
                border-radius: 8px;
                margin: 0.8rem 0;
                display: flex;
                align-items: center;
            }

            .alert-error {
                background: #fef2f2;
                color: #991b1b;
                border: 1px solid #fecaca;
            }

            .alert i {
                margin-right: 0.5rem;
            }

            .footer {
                background: var(--light-bg);
                padding: 2rem;
                text-align: center;
                border-top: 1px solid var(--border-color);
                color: var(--text-secondary);
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1.5rem 0;
            }

            .stat-card {
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                box-shadow: var(--shadow);
                border: 1px solid var(--border-color);
                transition: all 0.3s ease;
                animation: fadeInUp 0.6s ease forwards;
                opacity: 0;
                transform: translateY(20px);
            }

            .stat-card:nth-child(1) { animation-delay: 0.1s; }
            .stat-card:nth-child(2) { animation-delay: 0.2s; }
            .stat-card:nth-child(3) { animation-delay: 0.3s; }
            .stat-card:nth-child(4) { animation-delay: 0.4s; }

            .stat-card:hover {
                transform: translateY(-10px) rotate(2deg);
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
            }

            .stat-icon {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                margin: 0 auto 1rem;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
                color: white;
                transition: all 0.3s ease;
                animation: iconFloat 3s ease-in-out infinite;
            }

            @keyframes iconFloat {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-5px); }
            }

            .stat-card:hover .stat-icon {
                transform: scale(1.1) rotate(360deg);
            }

            .stat-title {
                font-weight: 600;
                color: var(--text-primary);
                margin-bottom: 0.5rem;
            }

            .stat-desc {
                font-size: 0.875rem;
                color: var(--text-secondary);
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            <div class="header">
                <div class="header-content">
                    <h1><i class="fas fa-building"></i> Compression Cast Concrete (CCC) Design System</h1>
                    <div style="text-align: left; max-width: 800px; margin: 1.5rem auto 0; background: rgba(255,255,255,0.95); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; backdrop-filter: blur(10px); box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                        <p style="margin: 0; line-height: 1.6;">
                            <strong style="color: #1e40af; font-size: 1.2em; text-shadow: 0 1px 2px rgba(0,0,0,0.1);">About Compression Cast Concrete (CCC):</strong><br><br>
                            <span style="color: #374151; text-shadow: 0 1px 2px rgba(255,255,255,0.8);">Compression cast concrete (CCC) is a type of concrete produced using the compression casting technique (CCT), where pressure is applied to consolidate the mix instead of traditional vibration methods. This approach enhances compressive strength, elastic modulus, density, and microstructure of concrete, while improving durability and reducing both production costs and CO₂ emissions.</span>
                        </p>
                    </div>
                    
                    <!-- Images Section in Header -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; max-width: 700px; margin: 2rem auto 0; padding: 0;">
                        <div style="text-align: center;">
                            <img src="/static/imgtwo.png" alt="Mix Design Process" style="width: 100%; height: 250px; object-fit: contain; border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.3); transition: all 0.3s ease; border: 2px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.1); padding: 10px;" onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 12px 30px rgba(0,0,0,0.4)'" onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 8px 20px rgba(0,0,0,0.3)'">
                            <h3 style="margin-top: 1rem; color: #ffffff; font-size: 1rem; font-weight: 600; text-shadow: 0 2px 4px rgba(0,0,0,0.3); line-height: 1.4;">Compression Casting of Concrete Cylinder</h3>
                        </div>
                        <div style="text-align: center;">
                                                    <img src="/static/imgone.png" alt="Concrete Testing" style="width: 100%; height: 250px; object-fit: contain; border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.3); transition: all 0.3s ease; border: 2px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.1); padding: 10px;" onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 12px 30px rgba(0,0,0,0.4)'" onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 8px 20px rgba(0,0,0,0.3)'">

                            <h3 style="margin-top: 1rem; color: #ffffff; font-size: 1rem; font-weight: 600; text-shadow: 0 2px 4px rgba(0,0,0,0.3); line-height: 1.4;">Compression Casting of Precast Beam</h3>
                        </div>
                    </div>
                </div>
            </div>

            <div class="content">

                <!-- Tab Navigation -->
                <div class="tab-navigation">
                    <button class="tab-button active" onclick="switchTab('prediction')" id="prediction-tab">
                        <i class="fas fa-microscope"></i>
                        Predict Properties
                    </button>
                    <button class="tab-button" onclick="switchTab('design')" id="design-tab">
                        <i class="fas fa-cogs"></i>
                        Design Optimal Mix
                    </button>
                </div>

                <!-- Tab Content -->
                <div id="prediction-content" class="tab-content active">
                    <div class="section">
                        <div class="section-header">
                            <div class="section-icon prediction-icon">
                                <i class="fas fa-microscope"></i>
                            </div>
                            <div>
                                <h2>Predict Concrete Properties</h2>
                            </div>
                        </div>
                        <p>Enter your concrete mix components below to predict the mechanical properties of your concrete mixture:</p>
                        
                        <div class="info-note" style="background: #e0f2fe; border-left: 4px solid #0288d1; padding: 1rem; margin-bottom: 1.5rem; border-radius: 4px;">
                            <i class="fas fa-info-circle" style="color: #0288d1; margin-right: 0.5rem;"></i>
                            <small><strong>Note:</strong> Optional fields are currently disabled and use standard values. Focus on the main components for accurate predictions.</small>
                        </div>
                        
                        <div class="form-grid">
                            <div class="form-group">
                                <label><i class="fas fa-mountain"></i> Cement Content</label>
                                <input type="number" id="cement" class="form-input" min="150" max="700" value="400" placeholder="kg/m³">
                                <small style="color: var(--text-secondary);">Range: 150-700 kg/m³</small>
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-grip-horizontal"></i> Fine Aggregate</label>
                                <input type="number" id="fine_aggregate" class="form-input" min="400" max="900" value="735" placeholder="kg/m³">
                                <small style="color: var(--text-secondary);">Range: 400-900 kg/m³</small>
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-cubes"></i> Coarse Aggregate</label>
                                <input type="number" id="coarse_aggregate" class="form-input" min="700" max="1300" value="1102" placeholder="kg/m³">
                                <small style="color: var(--text-secondary);">Range: 700-1300 kg/m³</small>
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-tint"></i> Water Content</label>
                                <input type="number" id="water" class="form-input" min="100" max="400" value="160" placeholder="kg/m³">
                                <small style="color: var(--text-secondary);">Range: 100-400 kg/m³</small>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label><i class="fas fa-compress-arrows-alt"></i> Casting Pressure</label>
                            <input type="number" id="casting_pressure" class="form-input" min="0" max="15" value="0" step="0.1" placeholder="MPa">
                            <small style="color: var(--text-secondary);">Range: 0-15 MPa (0 = traditional vibration)</small>
                        </div>
                        
                        <button class="btn btn-success" onclick="predictProperties()">
                            <i class="fas fa-calculator"></i> Predict Properties
                        </button>
                        
                        <div id="prediction_results" class="results">
                            <h3><i class="fas fa-chart-bar"></i> Prediction Results</h3>
                            <div id="prediction_content"></div>
                        </div>
                    </div>
                </div>

                <div id="cost-content" class="tab-content">
                    <div class="section">
                        <div class="section-header">
                            <div class="section-icon" style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white; animation: iconPulse 2s ease-in-out infinite 1s;">
                                <i class="fas fa-calculator"></i>
                            </div>
                            <div>
                                <h2>Cost & CO₂ Calculator</h2>
                            </div>
                        </div>
                        <p>Calculate production costs and environmental impact for your concrete mix:</p>
                        
                        <div class="form-grid">
                            <div class="form-group">
                                <label><i class="fas fa-mountain"></i> Cement Content</label>
                                <input type="number" id="cost_cement" class="form-input" min="150" max="700" value="400" placeholder="kg/m³">
                                <small style="color: var(--text-secondary);">Range: 150-700 kg/m³</small>
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-grip-horizontal"></i> Fine Aggregate</label>
                                <input type="number" id="cost_fine_aggregate" class="form-input" min="400" max="900" value="735" placeholder="kg/m³">
                                <small style="color: var(--text-secondary);">Range: 400-900 kg/m³</small>
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-cubes"></i> Coarse Aggregate</label>
                                <input type="number" id="cost_coarse_aggregate" class="form-input" min="700" max="1300" value="1102" placeholder="kg/m³">
                                <small style="color: var(--text-secondary);">Range: 700-1300 kg/m³</small>
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-tint"></i> Water Content</label>
                                <input type="number" id="cost_water" class="form-input" min="100" max="400" value="160" placeholder="kg/m³">
                                <small style="color: var(--text-secondary);">Range: 100-400 kg/m³</small>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label><i class="fas fa-compress-arrows-alt"></i> Casting Pressure</label>
                            <input type="number" id="cost_casting_pressure" class="form-input" min="0" max="15" value="0" step="0.1" placeholder="MPa">
                            <small style="color: var(--text-secondary);">Range: 0-15 MPa</small>
                        </div>
                        
                        <button class="btn" style="background: linear-gradient(135deg, #f59e0b, #d97706);" onclick="calculateCostCO2()">
                            <i class="fas fa-calculator"></i> Calculate Cost & CO₂
                        </button>
                        
                        <div id="cost_results" class="results">
                            <h3><i class="fas fa-chart-pie"></i> Cost & Environmental Impact</h3>
                            <div id="cost_content"></div>
                        </div>
                    </div>
                </div>

                <div id="design-content" class="tab-content">
                    <div class="section">
                        <div class="section-header">
                            <div class="section-icon design-icon">
                                <i class="fas fa-cogs"></i>
                            </div>
                            <div>
                                <h2>Design Optimal Mix</h2>
                            </div>
                        </div>
                        <p>Specify your target strength and optimization preferences to get the best concrete mix designs for your project:</p>
                        
                        <div class="info-note" style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; margin-bottom: 1.5rem; border-radius: 4px;">
                            <i class="fas fa-lightbulb" style="color: #ffc107; margin-right: 0.5rem;"></i>
                            <small><strong>Tip:</strong> Adjust the weights to prioritize cost, environmental impact, or density. Higher weights mean higher importance in optimization.</small>
                        </div>
                        
                        <div class="form-group">
                            <label><i class="fas fa-bullseye"></i> Target Compressive Strength</label>
                            <input type="number" id="target_strength" class="form-input" min="20" max="100" value="53" placeholder="MPa">
                            <small style="color: var(--text-secondary);">Range: 20-100 MPa</small>
                        </div>
                        
                        <div class="form-grid">
                            <div class="form-group">
                                <label><i class="fas fa-dollar-sign"></i> Cost Weight</label>
                                <input type="number" id="cost_weight" class="form-input" min="0" max="1" value="0.333" step="0.001" placeholder="0-1">
                                <small style="color: var(--text-secondary);">Priority for cost minimization (0-1)</small>
                            </div>
                            <div class="form-group">
                                <label><i class="fas fa-leaf"></i> CO₂ Weight</label>
                                <input type="number" id="co2_weight" class="form-input" min="0" max="1" value="0.333" step="0.001" placeholder="0-1">
                                <small style="color: var(--text-secondary);">Priority for CO₂ reduction (0-1)</small>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label><i class="fas fa-weight"></i> Density Weight</label>
                            <input type="number" id="density_weight" class="form-input" min="0" max="1" value="0.334" step="0.001" placeholder="0-1">
                            <small style="color: var(--text-secondary);">Priority for density maximization (0-1)</small>
                        </div>
                        
                        <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #0ea5e9;">
                            <i class="fas fa-info-circle" style="color: #0ea5e9;"></i>
                            <strong>Note:</strong> All weight values must sum to 1.0.
                        </div>

                        <!-- Optional Cost & CO2 Factors Section -->
                        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; margin: 2rem 0; border: 1px solid #e9ecef;">
                            <h4 style="color: var(--text-primary); margin-bottom: 1rem; display: flex; align-items: center;">
                                <i class="fas fa-sliders-h" style="margin-right: 0.5rem; color: var(--primary-color);"></i>
                                Optional Cost & CO₂ Factors
                                <button type="button" onclick="toggleFactors()" id="toggle-factors-btn" style="margin-left: auto; background: var(--primary-color); color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; font-size: 0.85rem; cursor: pointer;">
                                    <i class="fas fa-chevron-down"></i> Show Factors
                                </button>
                            </h4>
                            
                            <div id="factors-section" style="display: none;">
                                <p style="color: var(--text-secondary); margin-bottom: 1.5rem; font-size: 0.9rem;">
                                    <i class="fas fa-info-circle" style="color: #17a2b8;"></i>
                                    Customize material costs, CO₂ emission factors, and transport distances. Default values are used if not specified.
                                </p>
                                
                                <div class="form-grid">
                                    <div class="form-group">
                                        <label><i class="fas fa-dollar-sign"></i> Cement Cost ($/kg)</label>
                                        <input type="number" id="cement_cost_factor" class="form-input" min="0" step="0.001" placeholder="0.0631 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: $0.0631/kg</small>
                                    </div>
                                    <div class="form-group">
                                        <label><i class="fas fa-dollar-sign"></i> Sand Cost ($/kg)</label>
                                        <input type="number" id="fine_agg_cost_factor" class="form-input" min="0" step="0.001" placeholder="0.021 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: $0.021/kg</small>
                                    </div>
                                    <div class="form-group">
                                        <label><i class="fas fa-dollar-sign"></i> Coarse Aggregate Cost ($/kg)</label>
                                        <input type="number" id="coarse_agg_cost_factor" class="form-input" min="0" step="0.001" placeholder="0.017 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: $0.017/kg</small>
                                    </div>
                                    <div class="form-group">
                                        <label><i class="fas fa-dollar-sign"></i> Water Cost ($/kg)</label>
                                        <input type="number" id="water_cost_factor" class="form-input" min="0" step="0.001" placeholder="0.000691 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: $0.000691/kg</small>
                                    </div>
                                </div>
                                
                                <div class="form-grid">
                                    <div class="form-group">
                                        <label><i class="fas fa-leaf"></i> Cement CO₂ Factor (kg CO₂/kg)</label>
                                        <input type="number" id="cement_co2_factor" class="form-input" min="0" step="0.001" placeholder="0.82 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: 0.82 kg CO₂/kg</small>
                                    </div>
                                    <div class="form-group">
                                        <label><i class="fas fa-leaf"></i> Sand CO₂ Factor (kg CO₂/kg)</label>
                                        <input type="number" id="fine_agg_co2_factor" class="form-input" min="0" step="0.001" placeholder="0.0036 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: 0.0036 kg CO₂/kg</small>
                                    </div>
                                    <div class="form-group">
                                        <label><i class="fas fa-leaf"></i> Coarse Aggregate CO₂ Factor (kg CO₂/kg)</label>
                                        <input type="number" id="coarse_agg_co2_factor" class="form-input" min="0" step="0.001" placeholder="0.007 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: 0.007 kg CO₂/kg</small>
                                    </div>
                                    <div class="form-group">
                                        <label><i class="fas fa-leaf"></i> Water CO₂ Factor (kg CO₂/kg)</label>
                                        <input type="number" id="water_co2_factor" class="form-input" min="0" step="0.001" placeholder="0.000181 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: 0.000181 kg CO₂/kg</small>
                                    </div>
                                </div>
                                
                                <div class="form-grid">
                                    <div class="form-group">
                                        <label><i class="fas fa-truck"></i> Cement Transport Distance (km)</label>
                                        <input type="number" id="cement_transport_distance" class="form-input" min="0" step="1" placeholder="120.9 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: 120.9 km</small>
                                    </div>
                                    <div class="form-group">
                                        <label><i class="fas fa-truck"></i> Sand Transport Distance (km)</label>
                                        <input type="number" id="fine_agg_transport_distance" class="form-input" min="0" step="1" placeholder="63.4 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: 63.4 km</small>
                                    </div>
                                    <div class="form-group">
                                        <label><i class="fas fa-truck"></i> Coarse Aggregate Transport Distance (km)</label>
                                        <input type="number" id="coarse_agg_transport_distance" class="form-input" min="0" step="1" placeholder="34.9 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: 34.9 km</small>
                                    </div>
                                    <div class="form-group">
                                        <label><i class="fas fa-truck"></i> Water Transport Distance (km)</label>
                                        <input type="number" id="water_transport_distance" class="form-input" min="0" step="1" placeholder="0 (default)" value="">
                                        <small style="color: var(--text-secondary);">Default: 0 km</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <button class="btn" onclick="designMix()">
                            <i class="fas fa-magic"></i> Optimize Mix Design
                        </button>
                        
                        <div id="design_results" class="results">
                            <h3><i class="fas fa-trophy"></i> Optimized Mix Designs</h3>
                            <div id="design_content"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="footer">
                <p><i class="fas fa-info-circle"></i> Advanced Concrete Mix Design System - Empowering sustainable construction through intelligent mix optimization</p>
            </div>
        </div>

        <script>
            // Tab switching functionality
            function switchTab(tabName) {
                // Hide all tab contents
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => {
                    content.classList.remove('active');
                });
                
                // Remove active class from all tab buttons
                const tabButtons = document.querySelectorAll('.tab-button');
                tabButtons.forEach(button => {
                    button.classList.remove('active');
                });
                
                // Show selected tab content
                document.getElementById(tabName + '-content').classList.add('active');
                
                // Add active class to selected tab button
                document.getElementById(tabName + '-tab').classList.add('active');
            }

            // Toggle factors section
            function toggleFactors() {
                const factorsSection = document.getElementById('factors-section');
                const toggleBtn = document.getElementById('toggle-factors-btn');
                
                if (factorsSection.style.display === 'none') {
                    factorsSection.style.display = 'block';
                    toggleBtn.innerHTML = '<i class="fas fa-chevron-up"></i> Hide Factors';
                    factorsSection.style.animation = 'fadeInUp 0.3s ease-out';
                } else {
                    factorsSection.style.display = 'none';
                    toggleBtn.innerHTML = '<i class="fas fa-chevron-down"></i> Show Factors';
                }
            }

            async function predictProperties() {
                const button = event.target;
                const originalText = button.innerHTML;
                button.innerHTML = '<i class="fas fa-spinner loading"></i> Predicting<div class="loading-dots"><span></span><span></span><span></span></div>';
                button.disabled = true;
                button.style.background = 'var(--secondary-color)';

                const data = {
                    cement: parseFloat(document.getElementById('cement').value),
                    fine_aggregate: parseFloat(document.getElementById('fine_aggregate').value),
                    coarse_aggregate: parseFloat(document.getElementById('coarse_aggregate').value),
                    water: parseFloat(document.getElementById('water').value),
                    casting_pressure: parseFloat(document.getElementById('casting_pressure').value)
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        document.getElementById('prediction_content').innerHTML = `
                            <div class="result-item">
                                <span class="result-label"><i class="fas fa-weight"></i> Density</span>
                                <span class="result-value">${result.density.toFixed(2)} kg/m³</span>
                            </div>
                            <div class="result-item">
                                <span class="result-label"><i class="fas fa-compress"></i> Compressive Strength</span>
                                <span class="result-value">${result.strength.toFixed(2)} MPa</span>
                            </div>
                            <div class="result-item">
                                <span class="result-label"><i class="fas fa-expand-arrows-alt"></i> Elastic Modulus</span>
                                <span class="result-value">${result.elastic_modulus.toFixed(2)} GPa</span>
                            </div>
                            <div class="result-item">
                                <span class="result-label"><i class="fas fa-arrows-alt-h"></i> Peak Strain</span>
                                <span class="result-value">${result.peak_strain.toFixed(4)}</span>
                            </div>
                        `;
                        document.getElementById('prediction_results').style.display = 'block';
                        
                        // Add success animation
                        button.style.background = 'var(--gradient-4)';
                        setTimeout(() => {
                            button.style.background = 'var(--gradient-4)';
                        }, 1000);
                    } else {
                        showError('Prediction failed: ' + result.detail);
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                } finally {
                    button.innerHTML = originalText;
                    button.disabled = false;
                    button.style.background = 'var(--gradient-4)';
                }
            }
            
            async function calculateCostCO2() {
                const button = event.target;
                const originalText = button.innerHTML;
                button.innerHTML = '<i class="fas fa-spinner loading"></i> Calculating<div class="loading-dots"><span></span><span></span><span></span></div>';
                button.disabled = true;
                button.style.background = 'var(--secondary-color)';

                const data = {
                    cement: parseFloat(document.getElementById('cost_cement').value),
                    fine_aggregate: parseFloat(document.getElementById('cost_fine_aggregate').value),
                    coarse_aggregate: parseFloat(document.getElementById('cost_coarse_aggregate').value),
                    water: parseFloat(document.getElementById('cost_water').value),
                    casting_pressure: parseFloat(document.getElementById('cost_casting_pressure').value)
                };
                
                try {
                    const response = await fetch('/calculate-cost-co2', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        document.getElementById('cost_content').innerHTML = `
                            <div class="result-item">
                                <span class="result-label"><i class="fas fa-dollar-sign"></i> Total Production Cost</span>
                                <span class="result-value">$${result.cost.toFixed(2)}</span>
                            </div>
                            <div class="result-item">
                                <span class="result-label"><i class="fas fa-leaf"></i> CO₂ Emissions</span>
                                <span class="result-value">${result.co2_emissions.toFixed(2)} kg</span>
                            </div>
                            <div style="margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 8px; border-left: 4px solid #0ea5e9;">
                                <i class="fas fa-info-circle" style="color: #0ea5e9;"></i>
                                <strong>💡 Cost Breakdown:</strong> Material costs, transportation, and compaction energy are included in the calculation.
                            </div>
                        `;
                        document.getElementById('cost_results').style.display = 'block';
                        
                        // Add success animation
                        button.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
                    } else {
                        showError('Cost calculation failed: ' + result.detail);
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                } finally {
                    button.innerHTML = originalText;
                    button.disabled = false;
                    button.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
                }
            }
            
            async function designMix() {
                const button = event.target;
                const originalText = button.innerHTML;
                button.innerHTML = '<i class="fas fa-spinner loading"></i> Optimizing<div class="loading-dots"><span></span><span></span><span></span></div>';
                button.disabled = true;
                button.style.background = 'var(--secondary-color)';

                const data = {
                    target_strength: parseFloat(document.getElementById('target_strength').value),
                    cost_weight: parseFloat(document.getElementById('cost_weight').value),
                    co2_weight: parseFloat(document.getElementById('co2_weight').value),
                    density_weight: parseFloat(document.getElementById('density_weight').value)
                };

                // Add optional cost and CO2 factors if provided
                const optionalFactors = {};
                
                const cementCost = document.getElementById('cement_cost_factor').value;
                if (cementCost) optionalFactors.cement_cost = parseFloat(cementCost);
                
                const fineAggCost = document.getElementById('fine_agg_cost_factor').value;
                if (fineAggCost) optionalFactors.fine_agg_cost = parseFloat(fineAggCost);
                
                const coarseAggCost = document.getElementById('coarse_agg_cost_factor').value;
                if (coarseAggCost) optionalFactors.coarse_agg_cost = parseFloat(coarseAggCost);
                
                const waterCost = document.getElementById('water_cost_factor').value;
                if (waterCost) optionalFactors.water_cost = parseFloat(waterCost);
                
                const cementCO2 = document.getElementById('cement_co2_factor').value;
                if (cementCO2) optionalFactors.cement_co2 = parseFloat(cementCO2);
                
                const fineAggCO2 = document.getElementById('fine_agg_co2_factor').value;
                if (fineAggCO2) optionalFactors.fine_agg_co2 = parseFloat(fineAggCO2);
                
                const coarseAggCO2 = document.getElementById('coarse_agg_co2_factor').value;
                if (coarseAggCO2) optionalFactors.coarse_agg_co2 = parseFloat(coarseAggCO2);
                
                const waterCO2 = document.getElementById('water_co2_factor').value;
                if (waterCO2) optionalFactors.water_co2 = parseFloat(waterCO2);
                
                const cementTransport = document.getElementById('cement_transport_distance').value;
                if (cementTransport) optionalFactors.cement_transport_distance = parseFloat(cementTransport);
                
                const fineAggTransport = document.getElementById('fine_agg_transport_distance').value;
                if (fineAggTransport) optionalFactors.fine_agg_transport_distance = parseFloat(fineAggTransport);
                
                const coarseAggTransport = document.getElementById('coarse_agg_transport_distance').value;
                if (coarseAggTransport) optionalFactors.coarse_agg_transport_distance = parseFloat(coarseAggTransport);
                
                const waterTransport = document.getElementById('water_transport_distance').value;
                if (waterTransport) optionalFactors.water_transport_distance = parseFloat(waterTransport);

                // Add optional factors to data if any are provided
                if (Object.keys(optionalFactors).length > 0) {
                    data.cost_co2_factors = optionalFactors;
                }
                
                // Validate weights sum to approximately 1
                const weightSum = data.cost_weight + data.co2_weight + data.density_weight;
                if (Math.abs(weightSum - 1.0) > 0.01) {
                    // Auto-normalize weights if they don't sum to 1.0
                    normalizeWeights();
                    
                    // Update the data with normalized values
                    data.cost_weight = parseFloat(document.getElementById('cost_weight').value);
                    data.co2_weight = parseFloat(document.getElementById('co2_weight').value);
                    data.density_weight = parseFloat(document.getElementById('density_weight').value);
                    
                    // Show info message about normalization
                    const infoDiv = document.createElement('div');
                    infoDiv.className = 'alert';
                    infoDiv.style.background = '#e0f2fe';
                    infoDiv.style.color = '#0288d1';
                    infoDiv.style.border = '1px solid #b3e5fc';
                    infoDiv.innerHTML = `<i class="fas fa-info-circle"></i> Weights have been automatically normalized to sum to 1.0 for optimal results.`;
                    infoDiv.style.animation = 'fadeInUp 0.5s ease';
                    
                    // Remove any existing alerts
                    document.querySelectorAll('.alert').forEach(alert => alert.remove());
                    
                    // Add the info at the top of the content
                    const content = document.querySelector('.content');
                    content.insertBefore(infoDiv, content.firstChild);
                    
                    // Auto-remove after 3 seconds
                    setTimeout(() => {
                        infoDiv.style.animation = 'fadeOut 0.5s ease-in-out forwards';
                        setTimeout(() => infoDiv.remove(), 500);
                    }, 3000);
                }
                
                try {
                    const response = await fetch('/design-mix', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        let html = `
                            <div style="overflow-x: auto;">
                                <table class="data-table">
                                    <thead>
                                        <tr>
                                            <th><i class="fas fa-trophy"></i> Rank</th>
                                            <th><i class="fas fa-mountain"></i> Cement<br><small>(kg/m³)</small></th>
                                            <th><i class="fas fa-grip-horizontal"></i> Fine Agg.<br><small>(kg/m³)</small></th>
                                            <th><i class="fas fa-cubes"></i> Coarse Agg.<br><small>(kg/m³)</small></th>
                                            <th><i class="fas fa-tint"></i> Water<br><small>(kg/m³)</small></th>
                                            <th><i class="fas fa-compress-arrows-alt"></i> Pressure<br><small>(MPa)</small></th>
                                            <th><i class="fas fa-compress"></i> Strength<br><small>(MPa)</small></th>
                                            <th><i class="fas fa-weight"></i> Density<br><small>(kg/m³)</small></th>
                                            <th><i class="fas fa-chart-line"></i> Elastic Modulus<br><small>(GPa)</small></th>
                                            <th><i class="fas fa-expand-arrows-alt"></i> Peak Strain<br><small>(×10⁻³)</small></th>
                                            <th><i class="fas fa-dollar-sign"></i> Cost<br><small>($/m³)</small></th>
                                            <th><i class="fas fa-leaf"></i> CO₂<br><small>(kg CO₂/m³)</small></th>
                                            <th><i class="fas fa-star"></i> TOPSIS Score</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                        `;
                        
                        result.forEach(mix => {
                            const rankBadge = mix.rank <= 3 ? '🥇' : mix.rank <= 5 ? '🥈' : '🥉';
                            html += `
                                <tr>
                                    <td>${rankBadge} ${mix.rank}</td>
                                    <td>${mix.cement.toFixed(1)}</td>
                                    <td>${mix.fine_aggregate.toFixed(1)}</td>
                                    <td>${mix.coarse_aggregate.toFixed(1)}</td>
                                    <td>${mix.water.toFixed(1)}</td>
                                    <td>${mix.casting_pressure.toFixed(1)}</td>
                                    <td>${mix.predicted_strength.toFixed(2)}</td>
                                    <td>${mix.predicted_density.toFixed(1)}</td>
                                    <td>${mix.predicted_elastic_modulus.toFixed(2)}</td>
                                    <td>${mix.predicted_peak_strain.toFixed(3)}</td>
                                    <td>$${mix.cost.toFixed(2)}</td>
                                    <td>${mix.co2_emissions.toFixed(2)}</td>
                                    <td>${mix.topsis_score.toFixed(3)}</td>
                                </tr>
                            `;
                        });
                        
                        html += `
                                    </tbody>
                                </table>
                            </div>
                            <div style="margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%); border-radius: 8px; border-left: 4px solid #0288d1; animation: fadeInUp 0.5s ease;">
                                <i class="fas fa-info-circle" style="color: #0288d1;"></i>
                                <strong>🎯 Top recommendation:</strong> Mix #1 with TOPSIS score of ${result[0].topsis_score.toFixed(3)} offers the best balance of your specified criteria.
                            </div>
                            <div style="margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border-radius: 8px; border-left: 4px solid #ffc107;">
                                <i class="fas fa-info-circle" style="color: #856404;"></i>
                                <strong>📊 Results include:</strong> Material quantities, predicted properties (strength, density, elastic modulus, peak strain), total cost (materials + transport + compaction), and CO₂ emissions (materials + transport + compaction).
                            </div>
                        `;
                        
                        document.getElementById('design_content').innerHTML = html;
                        document.getElementById('design_results').style.display = 'block';
                        
                        // Add success animation
                        button.style.background = 'var(--gradient-3)';
                        setTimeout(() => {
                            button.style.background = 'var(--gradient-3)';
                        }, 1000);
                    } else {
                        showError('Mix design failed: ' + result.detail);
                    }
                } catch (error) {
                    showError('Network error: ' + error.message);
                } finally {
                    button.innerHTML = originalText;
                    button.disabled = false;
                    button.style.background = 'var(--gradient-3)';
                }
            }

            function showError(message) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'alert alert-error';
                errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
                errorDiv.style.animation = 'shake 0.5s ease-in-out';
                
                // Remove any existing alerts
                document.querySelectorAll('.alert').forEach(alert => alert.remove());
                
                // Add the error at the top of the content
                const content = document.querySelector('.content');
                content.insertBefore(errorDiv, content.firstChild);
                
                // Auto-remove after 5 seconds
                setTimeout(() => {
                    errorDiv.style.animation = 'fadeOut 0.5s ease-in-out forwards';
                    setTimeout(() => errorDiv.remove(), 500);
                }, 5000);
            }

            // Add shake animation for errors
            const shakeKeyframes = `
                @keyframes shake {
                    0%, 100% { transform: translateX(0); }
                    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
                    20%, 40%, 60%, 80% { transform: translateX(5px); }
                }
                @keyframes fadeOut {
                    from { opacity: 1; transform: translateY(0); }
                    to { opacity: 0; transform: translateY(-20px); }
                }
            `;
            
            const styleSheet = document.createElement('style');
            styleSheet.textContent = shakeKeyframes;
            document.head.appendChild(styleSheet);

            // Normalize weights only when form is submitted (not automatically)
            function normalizeWeights() {
                const costWeight = parseFloat(document.getElementById('cost_weight').value) || 0;
                const co2Weight = parseFloat(document.getElementById('co2_weight').value) || 0;
                const densityWeight = parseFloat(document.getElementById('density_weight').value) || 0;
                
                const total = costWeight + co2Weight + densityWeight;
                if (total > 0) {
                    document.getElementById('cost_weight').value = (costWeight / total).toFixed(3);
                    document.getElementById('co2_weight').value = (co2Weight / total).toFixed(3);
                    document.getElementById('density_weight').value = (densityWeight / total).toFixed(3);
                }
            }

            // Remove automatic normalization - only normalize when submitting
            document.addEventListener('DOMContentLoaded', function() {
                // No automatic event listeners for weight normalization
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict", response_model=ConcretePrediction)
async def predict_concrete_properties(input_data: ConcreteInput):
    """Predict concrete properties from mix composition"""
    try:
        predictions = predict_properties(input_data)
        
        return ConcretePrediction(
            density=predictions['pc'],
            strength=predictions['fc'],
            elastic_modulus=predictions['Ec'],
            peak_strain=predictions['e']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/design-mix", response_model=List[MixDesignResult])
async def design_concrete_mix(input_data: MixDesignInput):
    """Design optimal concrete mix for target strength"""
    try:
        weights = {
            'cost': input_data.cost_weight,
            'co2': input_data.co2_weight,
            'density': input_data.density_weight
        }
        
        # Extract custom factors if provided
        custom_costs = None
        custom_co2 = None
        custom_transport = None
        
        if input_data.cost_co2_factors:
            factors = input_data.cost_co2_factors
            
            # Build custom cost dict if any cost values provided
            if any([factors.cement_cost, factors.fine_agg_cost, factors.coarse_agg_cost, factors.water_cost]):
                custom_costs = {}
                if factors.cement_cost is not None:
                    custom_costs['cement_cost'] = factors.cement_cost
                if factors.fine_agg_cost is not None:
                    custom_costs['fine_agg_cost'] = factors.fine_agg_cost
                if factors.coarse_agg_cost is not None:
                    custom_costs['coarse_agg_cost'] = factors.coarse_agg_cost
                if factors.water_cost is not None:
                    custom_costs['water_cost'] = factors.water_cost
            
            # Build custom CO2 dict if any CO2 values provided
            if any([factors.cement_co2, factors.fine_agg_co2, factors.coarse_agg_co2, factors.water_co2]):
                custom_co2 = {}
                if factors.cement_co2 is not None:
                    custom_co2['cement_co2'] = factors.cement_co2
                if factors.fine_agg_co2 is not None:
                    custom_co2['fine_agg_co2'] = factors.fine_agg_co2
                if factors.coarse_agg_co2 is not None:
                    custom_co2['coarse_agg_co2'] = factors.coarse_agg_co2
                if factors.water_co2 is not None:
                    custom_co2['water_co2'] = factors.water_co2
            
            # Build custom transport dict if any transport values provided
            if any([factors.cement_transport_distance, factors.fine_agg_transport_distance, 
                    factors.coarse_agg_transport_distance, factors.water_transport_distance]):
                custom_transport = {}
                if factors.cement_transport_distance is not None:
                    custom_transport['cement_transport_distance'] = factors.cement_transport_distance
                if factors.fine_agg_transport_distance is not None:
                    custom_transport['fine_agg_transport_distance'] = factors.fine_agg_transport_distance
                if factors.coarse_agg_transport_distance is not None:
                    custom_transport['coarse_agg_transport_distance'] = factors.coarse_agg_transport_distance
                if factors.water_transport_distance is not None:
                    custom_transport['water_transport_distance'] = factors.water_transport_distance
        
        top_X, top_F, top_scores = optimize_mix_design(input_data.target_strength, weights, 
                                                        custom_costs, custom_co2, custom_transport)
        
        results = []
        for i, (mix, obj, score) in enumerate(zip(top_X, top_F, top_scores)):
            # Get predictions for this mix
            preds = get_predictions_from_mix(mix)
            
            results.append(MixDesignResult(
                rank=i+1,
                cement=mix[0],
                fine_aggregate=mix[1],
                coarse_aggregate=mix[2],
                water=mix[3],
                casting_pressure=mix[4],
                cost=obj[0],
                co2_emissions=obj[1],
                predicted_density=-obj[2],  # Convert back from negative
                predicted_strength=preds['fc'],
                predicted_elastic_modulus=preds['Ec'],
                predicted_peak_strain=preds['e'],
                topsis_score=score
            ))
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mix design error: {str(e)}")

class CostCO2Input(BaseModel):
    cement: float = Field(..., ge=150, le=700, description="Cement content (kg/m³)")
    fine_aggregate: float = Field(..., ge=400, le=900, description="Fine aggregate content (kg/m³)")
    coarse_aggregate: float = Field(..., ge=700, le=1300, description="Coarse aggregate content (kg/m³)")
    water: float = Field(..., ge=100, le=400, description="Water content (kg/m³)")
    casting_pressure: float = Field(..., ge=0, le=15, description="Casting pressure (MPa)")

class CostCO2Result(BaseModel):
    cost: float
    co2_emissions: float

@app.post("/calculate-cost-co2", response_model=CostCO2Result)
async def calculate_cost_co2(input_data: CostCO2Input):
    """Calculate cost and CO2 emissions for a concrete mix"""
    try:
        mix_array = [input_data.cement, input_data.fine_aggregate, 
                    input_data.coarse_aggregate, input_data.water]
        cost = calculate_cost(mix_array, input_data.casting_pressure)
        co2 = calculate_co2(mix_array, input_data.casting_pressure)
        
        return CostCO2Result(
            cost=cost,
            co2_emissions=co2
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost/CO2 calculation error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(bpnn_models)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)

