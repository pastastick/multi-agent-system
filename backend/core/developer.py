from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from core.experiment import ASpecificExp

if TYPE_CHECKING:
    from core.scenario import Scenario


class Developer(ABC, Generic[ASpecificExp]):
    def __init__(self, scen: Scenario) -> None:
        self.scen: Scenario = scen

    @abstractmethod
    def develop(self, exp: ASpecificExp) -> ASpecificExp:
        """
        Task Generator should take in an experiment.

        Because the schedule of different tasks is crucial for the final performance
        due to it affects the learning process.

        """
        #   abstract: "develop" experiment
        #   untuk coder: convert rumus faktor → kode Python yang bisa dieksekusi
        #   untuk runner: jalankan kode → hitung faktor → backtest → return result
        
        error_message = "generate method is not implemented."
        raise NotImplementedError(error_message)
