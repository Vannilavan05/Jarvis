"""System Information Plugin for Jarvis.
Retrieves metrics regarding CPU, RAM, disk storage, battery, and OS environment.
"""

import platform
from typing import Any, Dict, List
import psutil

try:
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.security.permissions import PermissionLevel
from .base import BasePlugin, PluginExecutionResult


class SystemInfoPlugin(BasePlugin):
    """Monitors system hardware utilization and operating environment."""

    @property
    def plugin_id(self) -> str:
        return "system_info"

    @property
    def name(self) -> str:
        return "System Information"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_intents(self) -> List[str]:
        return ["system_info"]

    @property
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        return {"system_info": PermissionLevel.READ_ONLY}

    def execute(
        self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginExecutionResult:
        metric = parameters.get("metric", "all").lower()

        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)

        disk = psutil.disk_usage("/")
        disk_percent = disk.percent
        disk_free_gb = disk.free / (1024**3)

        battery = psutil.sensors_battery()
        battery_str = "not detected"
        if battery:
            plugged = "plugged in" if battery.power_plugged else "on battery"
            battery_str = f"{battery.percent:.0f}% ({plugged})"

        data = {
            "cpu_percent": cpu_percent,
            "memory_percent": mem_percent,
            "memory_used_gb": round(mem_used_gb, 1),
            "memory_total_gb": round(mem_total_gb, 1),
            "disk_percent": disk_percent,
            "disk_free_gb": round(disk_free_gb, 1),
            "battery": battery_str,
            "os": f"{platform.system()} {platform.release()}",
        }

        if metric == "cpu":
            msg = f"CPU utilization is currently at {cpu_percent:.0f} percent."
        elif metric == "memory":
            msg = f"Memory usage is at {mem_percent:.0f} percent ({mem_used_gb:.1f} of {mem_total_gb:.1f} gigabytes used)."
        elif metric == "disk":
            msg = f"Primary drive has {disk_free_gb:.1f} gigabytes free with {disk_percent:.0f} percent used."
        elif metric == "battery":
            msg = f"Battery status: {battery_str}."
        else:
            msg = (
                f"System status: CPU at {cpu_percent:.0f} percent, "
                f"Memory at {mem_percent:.0f} percent, "
                f"Disk at {disk_percent:.0f} percent with {disk_free_gb:.1f} gigabytes free."
            )

        return PluginExecutionResult(success=True, spoken_response=msg, data=data)
