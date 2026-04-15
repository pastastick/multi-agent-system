"""
mendefinisikan rantai class abstrak yang menjembatani core.proposal (sangat abstrak) 
dengan factors.proposal (implementasi konkret).
"""


from abc import abstractmethod
from pathlib import Path
from typing import Tuple

from jinja2 import Environment, StrictUndefined

from core.experiment import Experiment
from core.prompts import Prompts
from core.proposal import (
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    Scenario,
    Trace,
)
from llm.client import LocalLLMBackend

#* load prompt templates
prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


#* LLM generate hypothesis, dengan 3 target: "factors", "model tuning", "feature engineering and model building"
class LLMHypothesisGen(HypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)

    # The following methods are scenario related so they should be implemented in the subclass
    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str) -> Hypothesis: ...

    def gen(self, trace: Trace) -> Hypothesis:
        context_dict, json_flag = self.prepare_context(trace)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_gen"]["system_prompt"])
            .render(
                # factors atau model tunning atau model building (apa yang sedang di-generate)
                targets=self.targets,   
                
                # deskripsi lengkap skenario(market, data, interface)
                # TODO harusnya sudah include insight dari agent eksternal
                scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                
                # output format(JSON) yang diharapkan
                hypothesis_output_format=context_dict["hypothesis_output_format"],
                
                # spesifikasi detail hipotesis
                hypothesis_specification=context_dict["hypothesis_specification"],
            )
        )
        #* render user prompt: riwayat + feedback dari hipotesis sebelumnya
        # TODO seharusnya sudah include insight dari agent eksternal
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_gen"]["user_prompt"])
            .render(
                targets=self.targets,
                hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                RAG=context_dict["RAG"],
            )
        )

        resp = LocalLLMBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)

        # parse JSON response dari LLM -> object Hypothesis
        hypothesis = self.convert_response(resp)

        return hypothesis

# set target di LLMHypothesisGen = "factors"
class FactorHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "factors"

# set target di LLMHypothesisGen = "model tuning"
class ModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "model tuning"

# set target di LLMHypothesisGen = "feature engineering and model building"
class FactorAndModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "feature engineering and model building"


#* LLM convert hipotesis menjadi eksperimen(kode program), dengan 3 target: "factors", "model tuning", "feature engineering and model building"
class LLMHypothesis2Experiment(Hypothesis2Experiment[Experiment]):
    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str, trace: Trace) -> Experiment: ...

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        context, json_flag = self.prepare_context(hypothesis, trace)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                experiment_output_format=context["experiment_output_format"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=self.targets,
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                target_list=context["target_list"],
                RAG=context["RAG"], 
            )
        )

        resp = LocalLLMBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)
        return self.convert_response(resp, trace)


class FactorHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "factors"


class ModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "model tuning"


class FactorAndModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "feature engineering and model building"
