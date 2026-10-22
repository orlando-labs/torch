package torch.registry_cuda_remote_verification

from net.http import Client

export main

class CoreClient:
  def init(@origin, token_file):
    @token = fs.read_text(token_file).trim
    @http = Client(timeout: 30.0, redirects: :off)

  def request(method, path, data = null):
    headers = {authorization: "Bearer " + @token, "content-type": "application/json"}
    response = case method:
      when "GET": @http.get(@origin + path, headers: headers)
      when "PUT": @http.put(@origin + path, headers: headers, body: Json.generate(data))
      when "POST": @http.post(@origin + path, headers: headers, body: Json.generate(data))
      else: raise ArgumentError("unsupported method")
    body = response.body_text(limit: 16777216)
    raise IOError("Remote core HTTP #{response.status}: #{body}") unless response.status == 200
    Json.parse(body, duplicate_keys: :reject)

  def wait(path, terminal):
    1800.times:
      value = request("GET", path)
      return value if terminal.contains?(value[:state])
      task.sleep(1.0)
    raise IOError("Remote core operation timed out")

  def close!(): @http.close!

def main():
  parser = ArgParser(name: "Verify published CUDA package through remote core")
  parser.arg("--url", required: true)
  parser.arg("--token-file", required: true)
  parser.arg("--source", required: true)
  parser.arg("--lock", required: true)
  parser.arg("--output", required: true)
  parser.arg("--version", default: "0.1.2")
  parser.arg("--reuse-packages")
  parser.arg("--core-root")
  args = parser.parse_or_raise()
  raise ArgumentError("--reuse-packages requires --core-root") if args[:reuse_packages] and not args[:core_root]
  api = CoreClient(args[:url], args[:token_file])
  try:
    project = "cuda_registry_" + SecureRandom.hex(6)
    route = "/v1/projects/" + project
    source = fs.read_text(args[:source])
    lock = fs.read_text(args[:lock])
    raise ArgumentError("expected canonical package and native variant in lock") unless lock.contains?("slowpilot/torch") and lock.contains?("cu130-prebuilt-source-runtime") and lock.contains?("sha256:")
    document = {format: "sputnik-notebook", version: 1, title: "Published Torch CUDA", active_sheet: "main", sheets: [{id: "main", title: "CUDA", cells: [{id: "1", kind: "code", source: source, mode: "manual"}]}], modules: [], auto_imports: []}
    uploaded = api.request("PUT", route, {command_id: SecureRandom.hex(16), base_revision: null, document: Json.generate(document), files: {"sputnik.lock": Base64.encode(lock.bytes)}})
    settings = api.request("GET", route + "/worker-settings/main")
    api.request("PUT", route + "/worker-settings/main", {command_id: SecureRandom.hex(16), base_version: settings[:version], runtime_epoch: settings[:runtime_epoch], grants: ["ffi.call", "ffi.load"]})
    accepted = null
    if args[:reuse_packages]:
      # Reuse the first core's verified store when checking a second core on
      # this host. Run still verifies every artifact/native hash against lock.
      destination = args[:core_root] + "/projects/" + project + "/packages"
      raise IOError("New project already has a package store") if fs.exists?(destination)
      fs.mkdir_p(fs.Path(destination).parent)
      staged = destination + ".staging"
      system.command("/bin/cp", "-a", "--", args[:reuse_packages], staged).capture(timeout: 900.0)
      fs.rename(staged, destination)
      accepted = {id: "verified-local-reuse"}
    else:
      accepted = api.request("POST", route + "/dependencies", {command_id: SecureRandom.hex(16), action: "install_sputnik", source: "slowpilot/torch", version: args[:version], backend: "cuda", bundle: "runtime", allow_native: true, allow_native_large: true})
      installed = api.wait(route + "/dependencies/" + accepted[:id], ["succeeded", "failed", "interrupted"])
      raise IOError("Notebook package installation failed: #{Json.generate(installed)}") unless installed[:state] == "succeeded"
    inventory = api.request("GET", route + "/dependencies")
    installed_package = inventory[:packages].find: $it[:name] == "slowpilot/torch"
    raise IOError("Notebook inventory lacks exact CUDA variant") unless installed_package and installed_package[:variant] == "linux-x86_64-gnu-cu130-prebuilt-source-runtime" and lock.contains?(installed_package[:digest])
    accepted_run = api.request("POST", route + "/runs", {command_id: SecureRandom.hex(16), revision: uploaded[:revision], sheet_id: "main", cell_id: "1", action: "run"})
    result = api.wait("/v1/runs/" + accepted_run[:run_id], ["succeeded", "failed", "cancelled", "force-stopped", "interrupted"])
    raise IOError("CUDA notebook Run failed: #{Json.generate(result)}") unless result[:state] == "succeeded"
    value = result[:result][:cells][0][:result]
    raise IOError("CUDA notebook result was not verified") unless value.contains?("PASS CUDA") and value.contains?("2.14.0")
    report = {schema: "torch.registry-cuda-verification.v1", project: project, revision: uploaded[:revision], installation: accepted[:id], "package": installed_package, run: result, verified: true}
    fs.atomic_write_text(args[:output], Json.pretty_generate(report) + "\n")
    print Json.generate({verified: true, project: project, run_id: accepted_run[:run_id], variant: installed_package[:variant], digest: installed_package[:digest], value: value})
  ensure:
    api.close!
  0
