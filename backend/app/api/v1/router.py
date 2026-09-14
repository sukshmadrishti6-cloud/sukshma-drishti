from app.api.v1 import auth, baselines, benchmarks, comparison, diagnostics, diseases, evaluation, experiments, health, history, models, pipeline, predict, recommendations, report
from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(diagnostics.router, prefix="/diagnostics", tags=["Diagnostics"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(diseases.router, prefix="/diseases", tags=["Diseases"])
api_router.include_router(history.router, prefix="/history", tags=["Prediction History"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["Recommendations"])
api_router.include_router(predict.router, tags=["Prediction"])
api_router.include_router(report.router, prefix="/report", tags=["Medical Report Scanner"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
api_router.include_router(baselines.router, prefix="/baselines", tags=["Baselines"])
api_router.include_router(comparison.router, prefix="/comparison", tags=["Comparison"])
api_router.include_router(benchmarks.router, prefix="/benchmarks", tags=["Benchmarks"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])

