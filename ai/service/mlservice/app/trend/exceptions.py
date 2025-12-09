"""커스텀 예외 클래스 정의"""


class ModelNotFoundError(Exception):
    """모델을 찾을 수 없을 때 발생하는 예외"""
    pass


class DataLoadError(Exception):
    """데이터 로딩 실패 시 발생하는 예외"""
    pass


class TrainingError(Exception):
    """학습 중 오류 발생 시 예외"""
    pass


class PredictionError(Exception):
    """예측 중 오류 발생 시 예외"""
    pass


class ModelSaveError(Exception):
    """모델 저장 실패 시 예외"""
    pass


class InvalidInputError(Exception):
    """잘못된 입력 데이터 예외"""
    pass

