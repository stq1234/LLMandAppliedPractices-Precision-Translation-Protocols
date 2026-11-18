# 题库来源与许可说明

## 原创题库

`items.jsonl` 中的 40 条英文规范语句及其中文锚点均由本项目原创撰写，不是 RFC 原文摘录或逐句改写。题库按真实规范中常见的要求力度、参与者角色、条件与例外、边界数值、时序、缓存/安全约束以及语法注册等技术主题构造。每条的 `source_basis` 字段仅标明启发主题及参考章节；它不表示该条款来自被引 RFC，也不暗示 RFC 对本项目原创句子背书。

## 核验过的官方来源

- RFC 2119, *Key words for use in RFCs to Indicate Requirement Levels*, §1–§5，说明 MUST/SHOULD/MAY 等规范词的力度及可例外情况。RFC Editor 页面：<https://www.rfc-editor.org/rfc/rfc2119.html>。
- RFC 8174, *Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words*, 全文，澄清规范关键字的大写形式及引用 BCP 14 的用法。RFC Editor 页面：<https://www.rfc-editor.org/rfc/rfc8174.html>。
- RFC 9110, *HTTP Semantics*, §2.2、§2.5、§12，涉及参与者角色、协议版本和内容协商。RFC Editor 页面：<https://www.rfc-editor.org/rfc/rfc9110.html>。
- RFC 8446, *The Transport Layer Security (TLS) Protocol Version 1.3*, §2.3 与 §8，涉及握手完成状态、0-RTT 及应用约束。RFC Editor 页面：<https://www.rfc-editor.org/rfc/rfc8446.html>。
- RFC 7234, *Hypertext Transfer Protocol (HTTP/1.1): Caching*, §3–§5，涉及缓存可存储/可复用条件、陈旧响应及验证。它已被 RFC 9111 取代，仅作为题库构思时的历史机制来源。RFC Editor 页面：<https://www.rfc-editor.org/rfc/rfc7234.html>。
- RFC 5234, *Augmented BNF for Syntax Specifications: ABNF*, §2，涉及语法规则和生成/解析的语法边界。RFC Editor 页面：<https://www.rfc-editor.org/rfc/rfc5234.html>。
- RFC 8259, *The JavaScript Object Notation (JSON) Data Interchange Format*, §9 与 §10，涉及解析器接受语法、生成器及 interoperability。RFC Editor 页面：<https://www.rfc-editor.org/rfc/rfc8259.html>。

RFC Editor 说明 RFC 可在 IETF Trust 许可下下载、复制和分发，但许可也限制 IETF 标准流程外的修改；IETF Trust 的 FAQ 对非代码摘录和修改摘录另有说明。参见 [RFC Editor: How Can I Use RFCs?](https://www.rfc-editor.org/series/rfc-use/) 与 [IETF Trust: Reproducing RFCs FAQ](https://trustee.ietf.org/about/faq/)。本题库为避免复制或改写 RFC 文本，采用原创模拟语句；来源只用于核验技术主题和术语机制，不表示具体原创规则由所引 RFC 规定。来源状态于 2026-11-18 核验。
