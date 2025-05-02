import os
import argparse
from dotenv import load_dotenv
from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from mlProject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

def run_pipeline(stages=None):
    """
    Run the ML pipeline with specified stages.
    
    Args:
        stages (list): List of stage numbers to run. If None, run all stages.
    """
    # Load environment variables
    load_dotenv()
    
    # Define all pipeline stages
    pipeline_stages = [
        ("Data Ingestion", DataIngestionTrainingPipeline),
        ("Data Validation", DataValidationTrainingPipeline),
        ("Data Transformation", DataTransformationTrainingPipeline),
        ("Model Training", ModelTrainerTrainingPipeline),
        ("Model Evaluation", ModelEvaluationTrainingPipeline)
    ]
    
    # If stages are specified, filter the pipeline stages
    if stages:
        try:
            stages = [int(s) for s in stages]
            pipeline_stages = [pipeline_stages[i-1] for i in stages if 1 <= i <= len(pipeline_stages)]
        except (ValueError, IndexError):
            logger.error(f"Invalid stage numbers: {stages}. Using all stages.")
    
    # Run each stage
    for i, (stage_name, pipeline_class) in enumerate(pipeline_stages, 1):
        logger.info(f"Running stage {i}: {stage_name}")
        try:
            pipeline = pipeline_class()
            pipeline.main()
            logger.info(f"Stage {i}: {stage_name} completed successfully")
        except Exception as e:
            logger.error(f"Error in stage {i}: {stage_name} - {e}")
            logger.error("Pipeline execution stopped due to error")
            return False
    
    logger.info("Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML pipeline")
    parser.add_argument(
        "--stages", 
        nargs="+", 
        help="Specify which stages to run (1-5). If not specified, all stages will run."
    )
    
    args = parser.parse_args()
    run_pipeline(args.stages)
