import re


def query_matches_anchor(query: str, anchor_name: str) -> bool:
    """检查查询内容与锚点名称是否有字符级关联。

    用于过滤 L1 级别的降级误匹配：当 LLM 无法找到精确锚点时会选
    顶层宽泛节点（L1），此时锚点名与用户意图往往毫无关联。
    策略：提取锚点名中的中文 2-gram 和长度≥3 的英文词，
    若均不出现在 query 中则判定为无关。
    """
    if not anchor_name:
        return False
    zh_segs = re.findall(r'[\u4e00-\u9fff]+', anchor_name)
    for seg in zh_segs:
        for i in range(len(seg) - 1):
            if seg[i:i + 2] in query:
                return True
    en_words = re.findall(r'[A-Za-z0-9]+', anchor_name)
    for word in en_words:
        if len(word) >= 3 and word.lower() in query.lower():
            return True
    return False
