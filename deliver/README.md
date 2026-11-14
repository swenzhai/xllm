# T5QKV当前交付包

[T5QKV交付件.zip](T5QKV交付件.zip) 为按任务书四类目录组织的单包，包含设计、全部具体case及测试、原格式报告/原日志/真实截图和完整算子源码及实际安装包。[task_submission.zip](task_submission.zip) 严格只含任务书同名七件。

本轮实现85f8644918f08ff07b356de568aa1ed1548c3183，分支t5-qkv-task-submission。默认分配器10批独立确认每批全10项性能通过；007最差74.18185us（≤75），017最差74.19035us（≤77）。page1g仅单列对照。精度35/35、工程206/206、内存10/10通过，三份XLSX只改内容及截图，原格式保持。

设计文档也可直接查看[完整设计与23项checklist](../task_submission/0%20T5QKV拆分与缓存追加算子设计文档.md)，报告原件在[task_submission](../task_submission/)。验证与压缩包索引.json记录SHA、固定实现及当前验收范围。当前授权环境自验证通过，可提交验收复核；指定OS/driver、任务书表头/TBE口径及社区流程限制见步骤说明，不表示正式验收已通过。
