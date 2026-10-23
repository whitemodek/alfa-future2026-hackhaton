# Runtime security and failure boundaries

This project owns the agent backend: provider calls, tool dispatch, process supervision,
artifact validation, and recovery. Model quality and ML correctness belong to the ML
implementation and are outside this backend's guarantees.

## Trust model

The task, dataset cells, filenames, model responses, generated Python, and optional ML
plugin are untrusted inputs. Tool schemas constrain orchestration requests; they do
not make generated code safe. Instructions found in data must not override the
system prompt or request credentials, installation, network access, or host changes.

The LLM provider receives task text, selected metadata, and tool output included in
the conversation. Run only on data permitted to be sent to that provider. Credentials
belong in environment variables, never task files, prompts, ZIP files, or Git.

## Isolation requirement

**A Python subprocess is not a security sandbox.** Removing API credentials from its
environment reduces accidental exposure but does not isolate the child from files,
the network, other processes, inherited OS permissions, or the parent process. Code
can read files outside its working directory or launch other programs. A malicious
child can also modify candidate artifacts and reports before registration.

Run evaluation in an expendable container or dedicated VM with only the intended
task data mounted, no host credentials, restricted networking, a nonprivileged user,
and platform-enforced CPU, memory, PID, and disk quotas. The backend's process timeout
and output limits improve reliability; the evaluator supplies the security boundary.
Do not use the agent against adversarial data on a developer workstation with secrets.

## Review checklist

- Environment: child processes do not inherit provider keys, cloud tokens, or unrelated
  secrets. Only explicit runtime environment variables cross the process boundary.
- Logs: no Authorization headers, API keys, complete environment dumps, or provider
  error bodies containing credentials. Redaction must happen before persistence.
- Tools: reject unknown names, malformed arguments, nonfinite numeric limits, and
  unsupported fields. Bound generated source, tool output, and conversation size.
- Processes: use argument lists rather than a shell; enforce deadlines and terminate
  the process group/tree on cancellation. Preserve sufficient time to publish results.
- Artifacts: resolve paths inside the task/work directory, reject escapes and symlinks
  where relevant, validate CSV structure and finite values, and cap artifact sizes.
- Selection: compare only scores using the same metric, direction, and validation
  split. Reject NaN and infinity. Preserve a private best snapshot and atomically
  publish it; failed later experiments must not replace a valid result.
- Provider: bound calls, tokens, retries, and request timeouts. Treat credentials and
  authentication failures as terminal; account conservatively for ambiguous failures.
- Packaging: include only required root Python files and the organizer-approved
  requirements file. Exclude task data, credentials, temporary runs, logs, and caches.

## Independent backend verification

Exercise public interfaces using tiny synthetic CSVs and a deterministic fake LLM
provider. These checks validate orchestration, not ML performance:

1. An authorized tool sequence produces the expected submission and exit status.
2. A weaker or malformed candidate does not overwrite the previous best result.
3. Provider failure, retry exhaustion, step exhaustion, and runtime expiry recover the
   best registered result when one exists; no result produces an explicit failure.
4. Invalid tool calls and malformed JSON receive bounded, useful errors.
5. Generated code sees no provider credential; secrets are absent from persisted logs.
6. Child timeout/output overflow and path traversal cannot bypass backend validation.
7. Launching from an unrelated current directory still writes beside task.md.
8. The ML plugin runs in a supervised child process and returns bounded JSON using
   the documented request/response contract.

Passing these checks does not prove model accuracy, prevent malicious generated code,
or replace isolation and resource limits supplied by the execution platform.
