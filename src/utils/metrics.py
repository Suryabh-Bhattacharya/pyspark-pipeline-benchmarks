"""Unified Spark REST API Metrics Collector.

Extracts job execution metrics (shuffle, spill, task skew) by querying the
Spark UI REST API at http://localhost:4040/api/v1.
"""

from typing import Dict, Any, List
import requests


class MetricsCollector:
    """Extracts job execution metrics via Spark UI REST API."""

    def __init__(self, spark_ui_url: str = "http://localhost:4040/api/v1"):
        self.base_url = spark_ui_url

    def get_completed_stage_ids(self) -> List[int]:
        """Returns a list of currently completed stage IDs across the active application."""
        try:
            apps_resp = requests.get(f"{self.base_url}/applications", timeout=1.0)
            if apps_resp.status_code != 200 or not apps_resp.json():
                return []
            app_id = apps_resp.json()[0]["id"]

            stages_resp = requests.get(
                f"{self.base_url}/applications/{app_id}/stages", timeout=1.0
            )
            if stages_resp.status_code != 200 or not stages_resp.json():
                return []

            return [
                s["stageId"]
                for s in stages_resp.json()
                if s.get("status") == "COMPLETE"
            ]
        except Exception:
            return []

    def get_job_stage_metrics(
        self, stage_ids_before: List[int]
    ) -> Dict[str, Any]:
        """Aggregates metrics for all stages completed since `stage_ids_before` was captured."""
        fallback = {
            "shuffle_read_mb": 0.0,
            "shuffle_write_mb": 0.0,
            "memory_spill_mb": 0.0,
            "disk_spill_mb": 0.0,
            "skew_ratio": 1.0,
        }

        try:
            apps_resp = requests.get(f"{self.base_url}/applications", timeout=1.0)
            if apps_resp.status_code != 200 or not apps_resp.json():
                return fallback

            app_id = apps_resp.json()[0]["id"]
            stages_resp = requests.get(
                f"{self.base_url}/applications/{app_id}/stages", timeout=1.0
            )

            if stages_resp.status_code != 200 or not stages_resp.json():
                return fallback

            # Filter out stages completed before the benchmark run started
            new_stages = [
                s
                for s in stages_resp.json()
                if s["stageId"] not in stage_ids_before
                and s.get("status") == "COMPLETE"
            ]

            if not new_stages:
                new_stages = [stages_resp.json()[0]]

            total_shuffle_read = sum(
                s.get("shuffleReadBytes", 0) for s in new_stages
            )
            total_shuffle_write = sum(
                s.get("shuffleWriteBytes", 0) for s in new_stages
            )
            total_mem_spill = sum(
                s.get("memoryBytesSpilled", 0) for s in new_stages
            )
            total_disk_spill = sum(
                s.get("diskBytesSpilled", 0) for s in new_stages
            )

            # Calculate task skew ratio across all tasks in these stages
            all_durations = []
            for stage in new_stages:
                s_id = stage["stageId"]
                a_id = stage.get("attemptId", 0)
                detail_resp = requests.get(
                    f"{self.base_url}/applications/{app_id}/stages/{s_id}/{a_id}",
                    timeout=1.0,
                )
                if detail_resp.status_code == 200:
                    tasks = detail_resp.json().get("tasks", {}).values()
                    all_durations.extend(
                        [t.get("duration", 0) for t in tasks if "duration" in t]
                    )

            skew_ratio = 1.0
            if all_durations:
                avg_dur = sum(all_durations) / len(all_durations)
                skew_ratio = round(max(all_durations) / max(avg_dur, 1.0), 2)

            return {
                "shuffle_read_mb": round(total_shuffle_read / (1024 * 1024), 2),
                "shuffle_write_mb": round(total_shuffle_write / (1024 * 1024), 2),
                "memory_spill_mb": round(total_mem_spill / (1024 * 1024), 2),
                "disk_spill_mb": round(total_disk_spill / (1024 * 1024), 2),
                "skew_ratio": max(skew_ratio, 1.0),
            }

        except Exception:
            return fallback

    def get_latest_stage_metrics(self) -> Dict[str, Any]:
        """Legacy fallback to fetch metrics only from the most recent completed stage."""
        return self.get_job_stage_metrics(stage_ids_before=[])


# Backward compatibility alias
SparkMetricsCollector = MetricsCollector