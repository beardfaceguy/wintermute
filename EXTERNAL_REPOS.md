# External Repositories

This workspace contains several external git repositories cloned alongside the
main `wintermute` repo. After cloning wintermute itself, run the commands below
to reproduce the full workspace on a new machine.

## Prerequisites

Make sure your SSH keys are set up for GitHub on the new machine, or swap the
`git@github.com:` URLs for `https://github.com/` equivalents.

## Root repo

```bash
git clone git@github.com:beardfaceguy/wintermute.git
cd wintermute
```

## Initialize submodules

The repo declares a submodule for whisper.cpp:

```bash
git submodule update --init --recursive
```

This clones `https://github.com/ggerganov/whisper.cpp.git` into
`thVoice/models/whisper.cpp`.

## Clone external repos

```bash
git clone git@github.com:openclaw/openclaw.git openclaw
git clone git@github.com:lucidrains/titans-pytorch.git titans-pytorch
git clone git@github.com:awsdocs/aws-doc-sdk-examples.git aws-doc-sdk-examples
git clone git@github.com:awslabs/mcp.git mcp_aws
```

## One-liner

```bash
git clone git@github.com:beardfaceguy/wintermute.git && \
  cd wintermute && \
  git submodule update --init --recursive && \
  git clone git@github.com:openclaw/openclaw.git openclaw && \
  git clone git@github.com:lucidrains/titans-pytorch.git titans-pytorch && \
  git clone git@github.com:awsdocs/aws-doc-sdk-examples.git aws-doc-sdk-examples && \
  git clone git@github.com:awslabs/mcp.git mcp_aws
```

## Repo details

| Directory | Remote URL | Branch at time of snapshot |
|-----------|-----------|--------------------------|
| `openclaw/` | `git@github.com:openclaw/openclaw.git` | main |
| `titans-pytorch/` | `git@github.com:lucidrains/titans-pytorch.git` | main |
| `aws-doc-sdk-examples/` | `git@github.com:awsdocs/aws-doc-sdk-examples.git` | main |
| `mcp_aws/` | `git@github.com:awslabs/mcp.git` | main |
| `thVoice/models/whisper.cpp` | `https://github.com/ggerganov/whisper.cpp.git` | (submodule) |
