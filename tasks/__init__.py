from .node_level.prediction import DefaultNPTask, DefaultTextNPTask, SubgraphTextNPTask, SubgraphNPTask
from .node_level.qa import NQATask
from .link_level.prediction import DefaultLPTask, DefaultTextLPTask, SubgraphLPTask, SubgraphTextLPTask
from .link_level.qa import LQATask
from .graph_level.prediction import DefaultGPTask, DefaultTextGPTask
from .graph_level.qa import GQATask
from .base import BaseTask