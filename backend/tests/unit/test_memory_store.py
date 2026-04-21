"""测试 MemoryStore — 使用 tmp_path，无网络依赖。"""

import json
import os

import pytest

from memory.memory_store import MemoryStore, _EMPTY_MEMORY, _MAX_FACTS, _MIN_CONFIDENCE


def make_store(tmp_path) -> MemoryStore:
    return MemoryStore(memory_file=str(tmp_path / "memory.json"))


class TestMemoryStoreLoad:

    def test_load_nonexistent_returns_empty(self, tmp_path):
        """文件不存在时返回空结构（含所有必要 key）。"""
        store = make_store(tmp_path)
        data = store.load()
        assert "user" in data
        assert "history" in data
        assert "facts" in data
        assert data["facts"] == []
        assert data["user"]["workContext"]["summary"] == ""

    def test_load_returns_deep_copy(self, tmp_path):
        """load() 每次返回独立副本，修改不影响下次读取。"""
        store = make_store(tmp_path)
        d1 = store.load()
        d1["facts"].append({"id": "x", "content": "test", "confidence": 0.9})
        d2 = store.load()
        assert d2["facts"] == []

    def test_load_fills_missing_keys(self, tmp_path):
        """兼容旧格式：缺失字段用默认值补全。"""
        path = tmp_path / "memory.json"
        path.write_text(json.dumps({"facts": []}), encoding="utf-8")
        store = MemoryStore(memory_file=str(path))
        data = store.load()
        assert "user" in data
        assert "history" in data


class TestMemoryStoreSave:

    def test_save_and_load_roundtrip(self, tmp_path):
        """保存再读取，内容一致。"""
        store = make_store(tmp_path)
        payload = {
            "user": {"workContext": {"summary": "看网分析师"}, "topOfMind": {"summary": "fgOTN 容量"}},
            "history": {"recentMonths": {"summary": "近三个月分析了5个场景"}},
            "facts": [{"id": "abc", "content": "偏好折线图", "category": "preference", "confidence": 0.9}],
        }
        store.save(payload)
        loaded = store.load()
        assert loaded["user"]["workContext"]["summary"] == "看网分析师"
        assert loaded["facts"][0]["content"] == "偏好折线图"

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path):
        """原子写完成后，临时 .tmp 文件不残留。"""
        store = make_store(tmp_path)
        store.save({"user": {"workContext": {"summary": "x"}, "topOfMind": {"summary": ""}},
                    "history": {"recentMonths": {"summary": ""}}, "facts": []})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f"残留临时文件: {tmp_files}"

    def test_save_overwrites_previous(self, tmp_path):
        """多次 save，最终内容以最后一次为准。"""
        store = make_store(tmp_path)
        store.save({**dict(_EMPTY_MEMORY), "user": {"workContext": {"summary": "v1"}, "topOfMind": {"summary": ""}}})
        store.save({**dict(_EMPTY_MEMORY), "user": {"workContext": {"summary": "v2"}, "topOfMind": {"summary": ""}}})
        assert store.load()["user"]["workContext"]["summary"] == "v2"


class TestMemoryStoreMergeUpdate:

    def test_merge_deduplicates_facts(self, tmp_path):
        """相同 content 的 fact 不重复写入。"""
        store = make_store(tmp_path)
        store.save({**dict(_EMPTY_MEMORY), "facts": [
            {"id": "1", "content": "喜欢折线图", "category": "preference", "confidence": 0.9}
        ]})
        update = {"facts": [
            {"content": "喜欢折线图", "category": "preference", "confidence": 0.85}
        ]}
        result = store.merge_update(update)
        contents = [f["content"] for f in result["facts"]]
        assert contents.count("喜欢折线图") == 1

    def test_merge_filters_low_confidence(self, tmp_path):
        """confidence < _MIN_CONFIDENCE 的 fact 被丢弃。"""
        store = make_store(tmp_path)
        update = {"facts": [
            {"content": "低置信度信息", "confidence": _MIN_CONFIDENCE - 0.1},
            {"content": "高置信度信息", "confidence": 0.9},
        ]}
        result = store.merge_update(update)
        contents = [f["content"] for f in result["facts"]]
        assert "低置信度信息" not in contents
        assert "高置信度信息" in contents

    def test_merge_caps_at_max_facts(self, tmp_path):
        """超过 _MAX_FACTS 条时只保留置信度最高的 _MAX_FACTS 条。"""
        store = make_store(tmp_path)
        # 先写入 _MAX_FACTS 条
        existing = [{"id": str(i), "content": f"fact_{i}", "category": "context", "confidence": 0.5}
                    for i in range(_MAX_FACTS)]
        store.save({**dict(_EMPTY_MEMORY), "facts": existing})
        # 再 merge 5 条新的（高置信度）
        update = {"facts": [
            {"content": f"new_fact_{i}", "confidence": 0.95} for i in range(5)
        ]}
        result = store.merge_update(update)
        assert len(result["facts"]) == _MAX_FACTS

    def test_merge_sorts_by_confidence(self, tmp_path):
        """merge_update 后 facts 按置信度降序排列。"""
        store = make_store(tmp_path)
        update = {"facts": [
            {"content": "低", "confidence": 0.6},
            {"content": "高", "confidence": 0.95},
            {"content": "中", "confidence": 0.75},
        ]}
        result = store.merge_update(update)
        confidences = [f["confidence"] for f in result["facts"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_merge_updates_work_context(self, tmp_path):
        """merge_update 更新 user.workContext.summary。"""
        store = make_store(tmp_path)
        update = {"user": {"workContext": {"summary": "新摘要"}, "topOfMind": {"summary": ""}}}
        store.merge_update(update)
        assert store.load()["user"]["workContext"]["summary"] == "新摘要"

    def test_merge_skips_empty_content_facts(self, tmp_path):
        """content 为空的 fact 被跳过。"""
        store = make_store(tmp_path)
        update = {"facts": [
            {"content": "", "confidence": 0.9},
            {"content": "  ", "confidence": 0.9},
            {"content": "有内容", "confidence": 0.9},
        ]}
        result = store.merge_update(update)
        assert len(result["facts"]) == 1


class TestMemoryStoreClear:

    def test_clear_resets_to_empty(self, tmp_path):
        """clear() 后 load() 返回空结构。"""
        store = make_store(tmp_path)
        store.save({**dict(_EMPTY_MEMORY),
                    "user": {"workContext": {"summary": "有内容"}, "topOfMind": {"summary": ""}},
                    "facts": [{"id": "1", "content": "记住了", "confidence": 0.9}]})
        store.clear()
        data = store.load()
        assert data["user"]["workContext"]["summary"] == ""
        assert data["facts"] == []

    def test_clear_creates_file(self, tmp_path):
        """clear() 在文件不存在时也能正常创建空文件。"""
        store = make_store(tmp_path)
        assert not os.path.exists(str(tmp_path / "memory.json"))
        store.clear()
        assert os.path.exists(str(tmp_path / "memory.json"))


class TestFormatForInjection:

    def test_empty_memory_returns_empty_string(self, tmp_path):
        """空 memory 时 format_for_injection 返回空字符串。"""
        store = make_store(tmp_path)
        result = store.format_for_injection()
        assert result == ""

    def test_includes_work_context(self, tmp_path):
        store = make_store(tmp_path)
        store.save({**dict(_EMPTY_MEMORY),
                    "user": {"workContext": {"summary": "看网专家"}, "topOfMind": {"summary": ""}}})
        result = store.format_for_injection()
        assert "看网专家" in result

    def test_respects_max_tokens(self, tmp_path):
        """超出 max_tokens 时截断。"""
        store = make_store(tmp_path)
        long_summary = "x" * 10000
        store.save({**dict(_EMPTY_MEMORY),
                    "user": {"workContext": {"summary": long_summary}, "topOfMind": {"summary": ""}}})
        result = store.format_for_injection(max_tokens=100)
        assert len(result) <= 100 * 3 + 50  # 允许少量偏差
