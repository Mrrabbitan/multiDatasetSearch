"""
查询追踪与监控模块

功能：
1. 记录每次查询的完整链路（问题 → 意图识别 → SQL 生成 → 执行结果）
2. 记录错误信息和堆栈
3. 计算各环节耗时
4. 提供查询接口和统计分析
5. 支持持久化到数据库或文件
"""

import json
import sqlite3
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class TraceStep:
    """单个执行步骤的追踪信息"""
    step_name: str  # 步骤名称（如 "parse_question", "generate_sql", "execute_sql"）
    start_time: float  # 开始时间戳
    end_time: Optional[float] = None  # 结束时间戳
    duration_ms: Optional[float] = None  # 耗时（毫秒）
    status: str = "running"  # 状态：running, success, error
    input_data: Optional[Dict] = None  # 输入数据
    output_data: Optional[Dict] = None  # 输出数据
    error_message: Optional[str] = None  # 错误信息
    error_traceback: Optional[str] = None  # 错误堆栈

    def finish(self, status: str = "success", output_data: Optional[Dict] = None, error: Optional[Exception] = None):
        """标记步骤完成"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        if output_data:
            self.output_data = output_data
        if error:
            self.error_message = str(error)
            self.error_traceback = traceback.format_exc()


@dataclass
class QueryTrace:
    """完整查询的追踪信息"""
    trace_id: str = field(default_factory=lambda: str(uuid4()))  # 唯一追踪ID
    question: str = ""  # 用户问题
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())  # 查询时间
    user_id: Optional[str] = None  # 用户ID（可选）
    session_id: Optional[str] = None  # 会话ID（可选）

    # 执行步骤
    steps: List[TraceStep] = field(default_factory=list)

    # 最终结果
    intent: Optional[str] = None  # 查询意图
    sql: Optional[str] = None  # 生成的SQL
    sql_params: Optional[List] = None  # SQL参数
    result_count: Optional[int] = None  # 结果数量
    final_answer: Optional[Any] = None  # 最终答案

    # 状态与性能
    status: str = "running"  # 整体状态：running, success, error
    total_duration_ms: Optional[float] = None  # 总耗时
    error_message: Optional[str] = None  # 错误信息

    # 元数据
    metadata: Dict = field(default_factory=dict)  # 额外元数据

    def add_step(self, step_name: str, input_data: Optional[Dict] = None) -> TraceStep:
        """添加新步骤"""
        step = TraceStep(
            step_name=step_name,
            start_time=time.time(),
            input_data=input_data
        )
        self.steps.append(step)
        return step

    def finish(self, status: str = "success", error: Optional[Exception] = None):
        """标记查询完成"""
        self.status = status
        if error:
            self.error_message = str(error)

        # 计算总耗时
        if self.steps:
            first_step_start = self.steps[0].start_time
            last_step_end = max((s.end_time for s in self.steps if s.end_time), default=time.time())
            self.total_duration_ms = (last_step_end - first_step_start) * 1000

    def to_dict(self) -> Dict:
        """转换为字典（用于序列化）"""
        data = asdict(self)
        # 简化 steps 数据（移除过大的字段）
        if data.get("steps"):
            for step in data["steps"]:
                # 限制输入输出数据大小
                if step.get("input_data"):
                    step["input_data"] = self._truncate_data(step["input_data"])
                if step.get("output_data"):
                    step["output_data"] = self._truncate_data(step["output_data"])
        return data

    @staticmethod
    def _truncate_data(data: Any, max_length: int = 1000) -> Any:
        """截断过长的数据"""
        if isinstance(data, str) and len(data) > max_length:
            return data[:max_length] + "... (truncated)"
        if isinstance(data, dict):
            return {k: QueryTrace._truncate_data(v, max_length) for k, v in data.items()}
        if isinstance(data, list) and len(data) > 10:
            return data[:10] + ["... (truncated)"]
        return data

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class TraceManager:
    """追踪管理器 - 负责持久化和查询"""

    def __init__(self, db_path: Optional[Path] = None, enable_file_log: bool = False, log_dir: Optional[Path] = None):
        """
        初始化追踪管理器

        Args:
            db_path: SQLite数据库路径（用于持久化）
            enable_file_log: 是否启用文件日志
            log_dir: 日志文件目录
        """
        self.db_path = db_path
        self.enable_file_log = enable_file_log
        self.log_dir = Path(log_dir) if log_dir else Path("poc/logs/traces")

        if self.db_path:
            self._init_db()

        if self.enable_file_log:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_traces (
                trace_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                intent TEXT,
                sql TEXT,
                sql_params TEXT,
                result_count INTEGER,
                status TEXT,
                total_duration_ms REAL,
                error_message TEXT,
                trace_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON query_traces(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON query_traces(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id ON query_traces(user_id)
        """)
        conn.commit()
        conn.close()

    def save_trace(self, trace: QueryTrace):
        """保存追踪记录"""
        # 保存到数据库
        if self.db_path:
            self._save_to_db(trace)

        # 保存到文件
        if self.enable_file_log:
            self._save_to_file(trace)

    def _save_to_db(self, trace: QueryTrace):
        """保存到数据库"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO query_traces
            (trace_id, question, timestamp, user_id, session_id, intent, sql, sql_params,
             result_count, status, total_duration_ms, error_message, trace_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trace.trace_id,
            trace.question,
            trace.timestamp,
            trace.user_id,
            trace.session_id,
            trace.intent,
            trace.sql,
            json.dumps(trace.sql_params) if trace.sql_params else None,
            trace.result_count,
            trace.status,
            trace.total_duration_ms,
            trace.error_message,
            trace.to_json()
        ))
        conn.commit()
        conn.close()

    def _save_to_file(self, trace: QueryTrace):
        """保存到文件"""
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"trace_{date_str}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(trace.to_json() + "\n")

    def get_trace(self, trace_id: str) -> Optional[QueryTrace]:
        """根据ID获取追踪记录"""
        if not self.db_path:
            return None

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM query_traces WHERE trace_id = ?", (trace_id,)).fetchone()
        conn.close()

        if not row:
            return None

        # 从 trace_data 重建对象
        trace_data = json.loads(row["trace_data"])
        return QueryTrace(**trace_data)

    def query_traces(
        self,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """查询追踪记录"""
        if not self.db_path:
            return []

        where_clauses = []
        params = []

        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if user_id:
            where_clauses.append("user_id = ?")
            params.append(user_id)
        if start_time:
            where_clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where_clauses.append("timestamp <= ?")
            params.append(end_time)

        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM query_traces {where_sql} ORDER BY timestamp DESC LIMIT ?",
            params + [limit]
        ).fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_statistics(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict:
        """获取统计信息"""
        if not self.db_path:
            return {}

        where_clauses = []
        params = []

        if start_time:
            where_clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where_clauses.append("timestamp <= ?")
            params.append(end_time)

        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        stats = {}

        # 总查询数
        stats["total_queries"] = conn.execute(f"SELECT COUNT(*) as cnt FROM query_traces {where_sql}", params).fetchone()["cnt"]

        # 成功/失败数
        stats["success_count"] = conn.execute(f"SELECT COUNT(*) as cnt FROM query_traces {where_sql} AND status = 'success'", params).fetchone()["cnt"]
        stats["error_count"] = conn.execute(f"SELECT COUNT(*) as cnt FROM query_traces {where_sql} AND status = 'error'", params).fetchone()["cnt"]

        # 平均耗时
        avg_duration = conn.execute(f"SELECT AVG(total_duration_ms) as avg FROM query_traces {where_sql} AND total_duration_ms IS NOT NULL", params).fetchone()["avg"]
        stats["avg_duration_ms"] = round(avg_duration, 2) if avg_duration else 0

        # 按意图分组统计
        intent_stats = conn.execute(f"SELECT intent, COUNT(*) as cnt FROM query_traces {where_sql} GROUP BY intent", params).fetchall()
        stats["by_intent"] = {row["intent"]: row["cnt"] for row in intent_stats}

        conn.close()
        return stats


# 全局单例
_trace_manager: Optional[TraceManager] = None


def init_trace_manager(db_path: Optional[Path] = None, enable_file_log: bool = True, log_dir: Optional[Path] = None):
    """初始化全局追踪管理器"""
    global _trace_manager
    _trace_manager = TraceManager(db_path=db_path, enable_file_log=enable_file_log, log_dir=log_dir)


def get_trace_manager() -> Optional[TraceManager]:
    """获取全局追踪管理器"""
    return _trace_manager
