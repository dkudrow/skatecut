"""Motion-based editing for fixed-camera videos."""

__version__ = "0.1.0"

from .model import Config, RenderOptions
from .pipeline import AnalysisResult, ProcessingResult, analyze_video, process_video, render_video, replan_video
from .progress import ProgressEvent
from .sequences import SequenceFilterOptions

__all__ = ["Config", "RenderOptions", "AnalysisResult", "ProcessingResult", "ProgressEvent",
           "analyze_video", "render_video", "process_video", "replan_video", "SequenceFilterOptions"]

