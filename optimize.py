import optuna
from train import train_model
from model import get_vit_tiny_model
from utils import get_dataloaders, TrainingConfig

def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Final validation loss
    """
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    
    # Create configuration
    config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_checkpoints=False,  # Disable checkpointing during optimization
        save_best_model=False,
        save_final_model=False
    )
    
    # Get data loaders and model
    train_loader, val_loader, _ = get_dataloaders(batch_size=config.batch_size)
    model = get_vit_tiny_model(config.num_classes)
    
    # Train model and return validation loss
    val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        return_history=False
    )
    
    return val_loss

def optimize_hyperparameters(n_trials: int = 20) -> dict:
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        n_trials: Number of optimization trials to run
        
    Returns:
        Dictionary containing the best hyperparameters
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)
    
    return study.best_params

if __name__ == "__main__":
    # Run hyperparameter optimization
    best_params = optimize_hyperparameters(n_trials=20)
    
    # Train final model with best hyperparameters
    best_config = TrainingConfig(
        batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
        save_checkpoints=True,
        save_best_model=True,
        save_final_model=True
    )
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=best_config.batch_size)
    model = get_vit_tiny_model(best_config.num_classes)
    
    history, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=best_config,
        return_history=True
    ) 