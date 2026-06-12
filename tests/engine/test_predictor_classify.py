"""Tests for Predictor.predict_classify."""
from unittest.mock import MagicMock


def _mock_classify_result(top1: int, conf: float):
    """Build a fake classify result with .probs.top1 and .probs.top1conf."""
    res = MagicMock()
    res.probs = MagicMock()
    res.probs.top1 = top1
    res.probs.top1conf = MagicMock()
    res.probs.top1conf.item = lambda: conf
    return res


def test_predict_classify_returns_top1():
    fake_model = MagicMock()
    fake_model.names = {0: "cat", 1: "dog"}
    fake_model.predict = MagicMock(return_value=[_mock_classify_result(1, 0.87)])

    from src.engine.predictor import Predictor
    p = Predictor(fake_model)
    result = p.predict_classify("dummy.jpg", project_classes=["cat", "dog"])
    assert result == ("dog", 0.87)


def test_predict_classify_returns_none_when_empty():
    from src.engine.predictor import Predictor
    fake_model = MagicMock()
    fake_model.predict = MagicMock(return_value=[])
    p = Predictor(fake_model)
    assert p.predict_classify("dummy.jpg", project_classes=["cat"]) is None


def test_predict_classify_class_not_in_project_returns_none():
    """模型预测类别不在项目类别表里 → 返回 None。"""
    from src.engine.predictor import Predictor
    fake_model = MagicMock()
    fake_model.names = {0: "lion"}
    fake_model.predict = MagicMock(return_value=[_mock_classify_result(0, 0.9)])
    p = Predictor(fake_model)
    assert p.predict_classify("x.jpg", project_classes=["cat", "dog"]) is None


def test_predict_classify_normalizes_class_name():
    """大小写/空白差异不应阻止匹配。"""
    from src.engine.predictor import Predictor
    fake_model = MagicMock()
    fake_model.names = {0: "Cat"}  # 模型用首字母大写
    fake_model.predict = MagicMock(return_value=[_mock_classify_result(0, 0.95)])
    p = Predictor(fake_model)
    result = p.predict_classify("x.jpg", project_classes=["cat"])
    assert result == ("cat", 0.95)


def test_predict_classify_no_probs_returns_none():
    """模型输出无 probs 字段 → None（detect 模型用错路径）。"""
    from src.engine.predictor import Predictor
    fake_model = MagicMock()
    res = MagicMock()
    res.probs = None
    fake_model.predict = MagicMock(return_value=[res])
    p = Predictor(fake_model)
    assert p.predict_classify("x.jpg", project_classes=["cat"]) is None


def test_predict_classify_filter_to_project_false_returns_unknown_class():
    """filter_to_project=False 时即便预测的类不在 project_classes 也应返回原始类名。"""
    from src.engine.predictor import Predictor
    fake_model = MagicMock()
    fake_model.names = {0: "lion"}
    fake_model.predict = MagicMock(return_value=[_mock_classify_result(0, 0.9)])
    p = Predictor(fake_model)
    result = p.predict_classify(
        "x.jpg", project_classes=["cat", "dog"], filter_to_project=False,
    )
    assert result == ("lion", 0.9)


def test_predict_classify_filter_to_project_false_with_empty_project_classes():
    """filter_to_project=False + 空 project_classes 与 None 等价：返回原始类名。"""
    from src.engine.predictor import Predictor
    fake_model = MagicMock()
    fake_model.names = {0: "fox"}
    fake_model.predict = MagicMock(return_value=[_mock_classify_result(0, 0.5)])
    p = Predictor(fake_model)
    assert p.predict_classify(
        "x.jpg", project_classes=[], filter_to_project=False,
    ) == ("fox", 0.5)


def test_predict_classify_filter_to_project_true_default_keeps_old_behavior():
    """默认 filter_to_project=True；不在项目类的预测仍返回 None。"""
    from src.engine.predictor import Predictor
    fake_model = MagicMock()
    fake_model.names = {0: "lion"}
    fake_model.predict = MagicMock(return_value=[_mock_classify_result(0, 0.9)])
    p = Predictor(fake_model)
    assert p.predict_classify("x.jpg", project_classes=["cat"]) is None


def test_predict_classify_batch_uses_single_model_call_for_multiple_images():
    from src.engine.predictor import Predictor

    fake_model = MagicMock()
    fake_model.names = {0: "cat", 1: "dog"}
    fake_model.predict = MagicMock(return_value=[
        _mock_classify_result(0, 0.91),
        _mock_classify_result(1, 0.82),
    ])
    p = Predictor(fake_model)

    result = p.predict_classify_batch(
        ["a.jpg", "b.jpg"], project_classes=["cat", "dog"], filter_to_project=False,
    )

    assert result == [("cat", 0.91), ("dog", 0.82)]
    fake_model.predict.assert_called_once_with(
        source=["a.jpg", "b.jpg"], verbose=False,
    )
