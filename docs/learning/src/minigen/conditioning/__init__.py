"""在历史课件中同时提供本目录的教学代码和共用工程组件。"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
