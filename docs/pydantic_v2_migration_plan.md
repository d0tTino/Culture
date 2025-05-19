# Pydantic v2 Migration Plan for Culture.ai

## 1. In-Depth Research of Pydantic v2 Migration

### Official Resources
- **Pydantic v2 Migration Guide:**  
  https://docs.pydantic.dev/latest/migration/
- **bump-pydantic (codemod tool):**  
  https://github.com/pydantic/bump-pydantic

### Community Guides & Insights
- **Qargo's Migration Experience:**  
  https://medium.com/codex/migrating-to-pydantic-v2-5a4b864621c3  
  - *Key insight:* Migrating incrementally is possible, but mixing v1/v2 models is risky and can cause subtle runtime errors.
  - *Pitfalls:* `each_item=True` in validators is gone; `json_encoders` replaced by field serializers; Config class replaced by `model_config`.
- **Workflow Orchestrator Guide:**  
  https://workfloworchestrator.org/orchestrator-core/migration-guide/2.0/  
  - *Key insight:* Use `bump-pydantic` for initial conversion, but expect manual fixes. Constrained types now use `Annotated`.

---

## 2. Comprehensive Audit of Pydantic v1 Usage in Culture.ai

### Key Files & Features Used

- **src/agents/graphs/basic_agent_graph.py**
  - Uses: `BaseModel`, `Field`, `create_model`
  - Purpose: Defines structured LLM outputs, agent turn state, and dynamic models for agent actions.

- **src/infra/llm_client.py**
  - Uses: `BaseModel`, `Field`, `ValidationError`, generic typing with Pydantic models, `.model_json_schema()`
  - Purpose: LLM client, structured output parsing, validation, and error handling.

- **src/infra/config.py**
  - Uses: None directly, but may use Pydantic for config models in the future.

- **src/infra/warning_filters.py**
  - Uses: Pydantic warning classes for filtering deprecation warnings.

- **DSPy program signatures and agent state models**
  - Likely use Pydantic for input/output field definitions and agent state serialization.

- **Tests**
  - Use Pydantic models for LLM output validation, test fixtures, and structured test data.

---

## 3. Detailed Mapping of Required Code Changes

### Mapping Table

| Pydantic v1 Pattern                | Pydantic v2 Equivalent/Change                |
|------------------------------------|----------------------------------------------|
| `class Config:`                    | `model_config = {}` (dict at class level)    |
| `@validator`                       | `@field_validator` (signature changes)       |
| `@root_validator`                  | `@model_validator` (signature changes)       |
| `Field(..., const=True)`           | Use `Literal` or `frozen=True`               |
| `allow_population_by_field_name`   | `populate_by_name` in `model_config`         |
| `.dict()`                          | `.model_dump()`                              |
| `.json()`                          | `.model_dump_json()`                         |
| `.schema()`                        | `.model_json_schema()`                       |
| `parse_obj_as`                     | Use `TypeAdapter`                            |
| `json_encoders` in Config          | Use `@field_serializer`                      |
| `conint`, `constr`, etc.           | Use `Annotated` with constraints             |
| `each_item=True` in validators     | Use `Annotated` and validate in item model   |

### Example Code Snippets

**Config Class → model_config**
```python
# Before
class MyModel(BaseModel):
    class Config:
        allow_population_by_field_name = True

# After
class MyModel(BaseModel):
    model_config = {"populate_by_name": True}
```

**Validator → Field Validator**
```python
# Before
@validator('name', pre=True)
def validate_name(cls, v):
    return v

# After
@field_validator('name', mode='before')
def validate_name(cls, v):
    return v
```

**dict() → model_dump()**
```python
# Before
data = model.dict(exclude_none=True)

# After
data = model.model_dump(exclude_none=True)
```

**schema() → model_json_schema()**
```python
# Before
schema = model.schema()

# After
schema = model.model_json_schema()
```

**Constrained Types**
```python
# Before
from pydantic import conint
age: conint(ge=0, le=120)

# After
from typing import Annotated
age: Annotated[int, Field(ge=0, le=120)]
```

---

## 4. Risk Assessment and Mitigation Strategies

### Dependency Compatibility

- **langchain, langgraph, dspy, chromadb, litellm**
  - **Action:** Check each library's release notes and issues for Pydantic v2 support.
  - **Mitigation:** If any library pins Pydantic v1, delay migration or isolate those dependencies. Use `pydantic.v1` imports as a temporary bridge if needed.

### Subtle Behavioral Changes

- **Stricter type coercion:** Pydantic v2 is less permissive; e.g., `Optional[str]` is required unless default is set.
- **Field constraints:** Some constraints (e.g., `regex`) now use `pattern`; `const` is replaced by `Literal`.
- **Validator signatures:** `@validator` and `@root_validator` are deprecated; signatures must be updated.

**Mitigation:**  
- Run all tests after migration.
- Manually test LLM output parsing and agent state transitions.

### Testing Impact

- **Tests using `.dict()`, `.json()`, or Pydantic validation will need updates.**
- **Mitigation:** Allocate time for test refactoring and validation.

### Migration Approach Risk

- **Incremental migration** is possible using `pydantic.v1` imports, but mixing v1/v2 models is risky (see Qargo's experience).
- **Mitigation:** Prefer a "big bang" migration on a feature branch, with all models and usages updated together.

---

## 5. Step-by-Step Migration Plan

### Phase 1: Preparation

- Ensure all tests pass on the main branch.
- Create a new branch: `feature/pydantic-v2-migration`.
- Install `bump-pydantic` for codemod support.

### Phase 2: Dependency Updates

- Update `pydantic` to latest v2.x in `requirements.txt` and `requirements-dev.txt`.
- Update `langchain`, `langgraph`, `dspy`, `chromadb`, `litellm` to Pydantic v2-compatible versions.
  - **Target versions:** (to be filled in after checking each library's docs)
- Run `pip install -r requirements.txt -r requirements-dev.txt` and resolve conflicts.

### Phase 3: Code Modifications

- **Automated pass:**  
  Run `bump-pydantic src/` and `bump-pydantic tests/`. Review all changes and TODOs.
- **Manual pass (module by module):**
  1. Update core models: `src/agents/core/agent_state.py`
  2. Update infrastructure: `src/infra/config.py`, `src/infra/llm_client.py`
  3. Update agent logic: `src/agents/graphs/basic_agent_graph.py`, DSPy programs
  4. Update tests
- After each module, run relevant unit tests.

### Phase 4: Testing and Validation

- Run the full test suite: `python -m pytest tests/ -v`
- Fix any test failures or validation errors.
- Manually test:
  - Agent initialization and state
  - LLM structured output parsing
  - Memory storage/retrieval
  - Configuration loading

### Phase 5: Documentation and Review

- Update internal docs and code comments referencing Pydantic v1.
- Prepare a PR for `feature/pydantic-v2-migration` with a summary of changes and critical review areas.
- **Rollback:** If major issues arise, abandon the feature branch and revert.

---

## 6. Effort Estimation

- **Preparation & dependency checks:** 0.5 day
- **Automated codemod pass:** 0.5 day
- **Manual code migration (core, infra, agent, tests):** 2–3 days
- **Testing, bugfixes, and validation:** 1–2 days
- **Documentation and PR review:** 0.5 day

**Total estimate:** 4–6 developer-days

---

## 7. Deliverable

**File:** `docs/pydantic_v2_migration_plan.md`  
**Contents:**  
- This plan, including all mappings, code snippets, risk analysis, and step-by-step instructions.

---

## 8. Completion Status and Key Learnings (2025-05-16)

### Migration Status
- **All core and relevant files in src/ and tests/ are now Pydantic v2 compliant.**
- **All unit and integration tests pass.**
- **No Pydantic v1 deprecation warnings remain from our codebase; only third-party library warnings persist.**
- **test_llm_monitoring.py was updated to ensure pytest test discovery (function renamed to test_llm_monitoring, model_config added).**
- **No direct Pydantic models or v1 patterns found in vector_store.py or warning_filters.py.**
- **archives/ files were not migrated, as they are not in active use.**
- **Documentation and code snippets in docs/ now reflect Pydantic v2 patterns.**

### Key Learnings & Challenges
- The codemod tool (`bump-pydantic`) was helpful for scanning, but manual review was essential for catching all v1 patterns and ensuring test coverage.
- Most migration effort was spent on model config, type hints, and test model usage.
- Pytest test discovery requires test functions to be prefixed with `test_` (not just marked with @pytest.mark.unit).
- Third-party libraries (e.g., discord.py, weaviate) may still emit v1 deprecation warnings, but these are outside our control.
- No breaking changes were required for vector store or warning filter modules.
- All code and documentation are now consistent with Pydantic v2 best practices.

### Final Verification
- **Command:** `python -m pytest tests/ -v -W default::DeprecationWarning --tb=short`
- **Result:** All 35 tests passed, no v1 warnings from our codebase.

---

**Task 111: Pydantic v2 Migration is now complete.** 