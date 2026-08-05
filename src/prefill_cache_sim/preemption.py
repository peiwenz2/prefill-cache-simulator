"""Cooperative decode-preemption protocol and safety ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class DecodeAction(StrEnum):
    REPORT_CHECKPOINT = "REPORT_CHECKPOINT"
    ABORT_SELF = "ABORT_SELF"


@dataclass(frozen=True, slots=True)
class DecodeCheckpoint:
    logical_request_id: str
    epoch: int
    output_seq: int
    emitted_tokens: int
    kv_handle: str | None
    tpot_work: float
    served_quantum: int

    def __post_init__(self) -> None:
        if not self.logical_request_id or self.epoch < 0 or self.output_seq < 0:
            raise ValueError("checkpoint identity and fence must be valid")
        if self.emitted_tokens < 0 or self.tpot_work <= 0 or self.served_quantum <= 0:
            raise ValueError("checkpoint progress must be positive")


@dataclass(frozen=True, slots=True)
class ActionGrant:
    grant_id: str
    logical_request_id: str
    action: DecodeAction
    source_epoch: int
    target_epoch: int


@dataclass(frozen=True, slots=True)
class KeepMoveInput:
    completion_value: float
    keep_probability: float
    move_probability: float
    remaining_gpu_work: float
    interference_gpu_work: float
    checkpoint_gpu_work: float
    recovery_gpu_work: float
    migration_stall_work: float
    duplicate_risk: float
    wait_work: float
    tenant_weight: float

    def __post_init__(self) -> None:
        probabilities = (self.keep_probability, self.move_probability)
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("completion probabilities must be in [0, 1]")
        if self.completion_value < 0 or self.tenant_weight <= 0:
            raise ValueError("value and tenant weight are invalid")
        work = (
            self.remaining_gpu_work,
            self.interference_gpu_work,
            self.checkpoint_gpu_work,
            self.recovery_gpu_work,
            self.migration_stall_work,
            self.duplicate_risk,
            self.wait_work,
        )
        if any(value < 0 for value in work):
            raise ValueError("preemption work inputs must be non-negative")


@dataclass(frozen=True, slots=True)
class PreemptionConfig:
    min_quantum: int = 14
    max_preemptions: int = 2
    safety_margin: float = 0.1
    lambda_gpu: float = 1.0
    lambda_interference: float = 1.0
    lambda_stall: float = 1.0
    lambda_risk: float = 1.0
    aging_rate: float = 0.01
    max_preemptions_per_completed: float = 0.25


@dataclass(frozen=True, slots=True)
class Decision:
    action: DecodeAction
    keep_score: float
    move_score: float
    reason: str


class R2CheckpointStore:
    """Separate epoch-fenced store; intentionally not a Mooncake prefix store."""

    def __init__(self) -> None:
        self._objects: dict[str, DecodeCheckpoint] = {}

    def put(self, checkpoint: DecodeCheckpoint) -> None:
        if checkpoint.kv_handle is None:
            raise ValueError("R2 checkpoint requires a KV handle")
        current = self._objects.get(checkpoint.kv_handle)
        if current is not None and (checkpoint.epoch, checkpoint.output_seq) <= (
            current.epoch,
            current.output_seq,
        ):
            raise ValueError("checkpoint store update is stale")
        self._objects[checkpoint.kv_handle] = checkpoint

    def get(self, handle: str) -> DecodeCheckpoint | None:
        return self._objects.get(handle)


class OutputAuthority:
    def __init__(self, epoch: int = 0, next_seq: int = 0) -> None:
        self.epoch = epoch
        self.next_seq = next_seq

    def advance_epoch(self, epoch: int, *, resume_output_seq: int) -> None:
        if epoch <= self.epoch:
            raise ValueError("output epoch must advance")
        self.epoch = epoch
        self.next_seq = max(self.next_seq, resume_output_seq)

    def accept(self, *, epoch: int, output_seq: int) -> bool:
        if epoch != self.epoch or output_seq != self.next_seq:
            return False
        self.next_seq += 1
        return True


class CooperativePreemptionController:
    def __init__(
        self,
        config: PreemptionConfig,
        checkpoint_store: R2CheckpointStore,
        output_authorities: dict[str, OutputAuthority] | None = None,
    ) -> None:
        self.config = config
        self.checkpoint_store = checkpoint_store
        self._preemptions: dict[str, int] = {}
        self._latest_epoch: dict[str, int] = {}
        self._authorities = output_authorities if output_authorities is not None else {}
        self._grant_sequence = 0

    @property
    def preemptions(self) -> MappingProxyType[str, int]:
        return MappingProxyType(dict(self._preemptions))

    def report_checkpoint(
        self,
        checkpoint: DecodeCheckpoint,
        inputs: KeepMoveInput,
        *,
        planner_available: bool = True,
    ) -> tuple[Decision, ActionGrant]:
        expected_epoch = self._latest_epoch.get(checkpoint.logical_request_id, 0)
        if checkpoint.epoch != expected_epoch:
            raise ValueError("checkpoint epoch is not controller-authorized")
        decision = self._decide(checkpoint, inputs, planner_available)
        target_epoch = checkpoint.epoch
        if decision.action is DecodeAction.ABORT_SELF:
            target_epoch += 1
            authority = self._authorities.get(checkpoint.logical_request_id)
            if authority is None:
                authority = OutputAuthority(checkpoint.epoch, checkpoint.output_seq)
            if authority.epoch != checkpoint.epoch:
                raise ValueError("output authority epoch diverged")
            if checkpoint.kv_handle is not None:
                self.checkpoint_store.put(checkpoint)
            self._authorities[checkpoint.logical_request_id] = authority
            authority.advance_epoch(
                target_epoch, resume_output_seq=checkpoint.output_seq
            )
            self._latest_epoch[checkpoint.logical_request_id] = target_epoch
            self._preemptions[checkpoint.logical_request_id] = (
                self._preemptions.get(checkpoint.logical_request_id, 0) + 1
            )
        else:
            self._latest_epoch[checkpoint.logical_request_id] = checkpoint.epoch
        self._grant_sequence += 1
        grant = ActionGrant(
            f"grant-{self._grant_sequence}",
            checkpoint.logical_request_id,
            decision.action,
            checkpoint.epoch,
            target_epoch,
        )
        return decision, grant

    def _decide(
        self,
        checkpoint: DecodeCheckpoint,
        inputs: KeepMoveInput,
        planner_available: bool,
    ) -> Decision:
        if not planner_available:
            return Decision(DecodeAction.REPORT_CHECKPOINT, 0, 0, "planner_fail_open")
        if checkpoint.served_quantum < self.config.min_quantum:
            return Decision(DecodeAction.REPORT_CHECKPOINT, 0, 0, "minimum_quantum")
        count = self._preemptions.get(checkpoint.logical_request_id, 0)
        if count >= self.config.max_preemptions:
            return Decision(DecodeAction.REPORT_CHECKPOINT, 0, 0, "bounded_preemption")
        protected_keep_value = inputs.completion_value + (
            self.config.aging_rate * inputs.wait_work * inputs.tenant_weight
        )
        keep = protected_keep_value * inputs.keep_probability - (
            self.config.lambda_gpu * inputs.remaining_gpu_work
            + self.config.lambda_interference * inputs.interference_gpu_work
        )
        move = inputs.completion_value * inputs.move_probability - (
            self.config.lambda_gpu
            * (inputs.checkpoint_gpu_work + inputs.recovery_gpu_work)
            + self.config.lambda_stall * inputs.migration_stall_work
            + self.config.lambda_risk * inputs.duplicate_risk
        )
        if move - keep > self.config.safety_margin:
            return Decision(DecodeAction.ABORT_SELF, keep, move, "move_exceeds_margin")
        return Decision(DecodeAction.REPORT_CHECKPOINT, keep, move, "keep_with_margin")


def preemption_gate(
    *, completed_requests: int, preemptions: int, config: PreemptionConfig
) -> bool:
    if completed_requests <= 0:
        return False
    return preemptions / completed_requests <= config.max_preemptions_per_completed
