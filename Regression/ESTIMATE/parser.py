from dataclasses import dataclass, field


@dataclass
class IndicatorConfig:
    medium_period: int = 10
    slow_period: int = 20


@dataclass
class DataConfig:
    symbols: list = field(default_factory=list)
    start_train: str = "2015-01-01"
    end_train: str = "2018-12-31"
    start_valid: str = "2019-01-10"
    end_valid: str = "2019-12-31"
    start_test: str = "2020-01-10"
    end_test: str = "2020-12-31"
    n_step_ahead: int = 5
    target_col: str = "trend_return"
    include_target: bool = False
    history_window: int = 20
    outlier_threshold: float = 1000
    indicators: dict = field(default_factory=dict)
    universe: str = "sp500"


@dataclass
class ModelConfig:
    path: str = "supervisor.Supervisor"
    confidence_threshold: float = 0.90
    earlystop: int = 16
    batch_size: int = 32
    epochs: int = 500
    hidden_dim: int = 32
    rnn_units: int = 16
    learning_rate: float = 0.0001
    cuda: bool = True
    resume: bool = False
    save_best: bool = True
    lr_decay: float = 0.005
    eval_iter: int = 10
    dropout: float = 0.4
    verify_threshold: float = 0.08
    model_name: str = "estimate"
    seed: int = 42


@dataclass
class BacktestConfig:
    config_path: str = "backtest/config/normal.yaml"


@dataclass
class ProjectConfig:
    run_name: str = "default_run"
    model_name: str = "estimate"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def build_config_from_args(args):
    # Indicatori
    indicators = {
        'close_sma': IndicatorConfig(args.close_sma_medium, args.close_sma_slow),
        'rsi': IndicatorConfig(args.rsi_medium, args.rsi_slow),
        'macd': IndicatorConfig(args.macd_medium, args.macd_slow),
        'mfi': IndicatorConfig(args.mfi_medium, args.mfi_slow),
        'trend_return': {}
    }

    data = DataConfig(
        symbols=args.symbols,
        start_train=args.start_train,
        end_train=args.end_train,
        start_valid=args.start_valid,
        end_valid=args.end_valid,
        start_test=args.start_test,
        end_test=args.end_test,
        n_step_ahead=args.n_step_ahead,
        target_col=args.target_col,
        include_target=args.include_target,
        history_window=args.history_window,
        outlier_threshold=args.outlier_threshold,
        indicators=indicators,
        universe=args.universe
    )

    model = ModelConfig(
        path=args.path,
        confidence_threshold=args.confidence_threshold,
        earlystop=args.earlystop,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        rnn_units=args.rnn_units,
        learning_rate=args.learning_rate,
        cuda=args.cuda,
        resume=args.resume,
        save_best=args.save_best,
        lr_decay=args.lr_decay,
        eval_iter=args.eval_iter,
        dropout=args.dropout,
        verify_threshold=args.verify_threshold,
        model_name=args.model,
        seed=args.seed
    )

    backtest = BacktestConfig(config_path=args.config_path)

    return ProjectConfig(
        run_name=args.run_name,
        model_name=args.model,
        data=data,
        model=model,
        backtest=backtest
    )
