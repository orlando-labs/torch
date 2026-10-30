package torch.registry_rocm_verification

from net.http import Client, Request

export main

class CoreClient:
  def init(@origin, token_file):
    @token = fs.read_text(token_file).trim
    @http = Client(timeout: 30.0, redirects: :off)

  def request(method, path, data = null):
    headers = {authorization: "Bearer " + @token, "content-type": "application/json"}
    body = Json.generate(data)
    response = case method:
      when "GET": @http.get(@origin + path, headers:)
      when "PUT": @http.put(@origin + path, headers:, body:)
      when "POST": @http.post(@origin + path, headers:, body:)
      when "DELETE": @http.send(Request(method: :delete, url: @origin + path, headers:, body:))
      else: raise ArgumentError("unsupported method")
    text = response.body_text(limit: 16777216)
    raise IOError("Remote HTTP #{response.status}: #{text}") unless response.status == 200
    Json.parse(text, duplicate_keys: :reject)

  def wait(path, terminal):
    180.times:
      value = request("GET", path)
      return value if value[:state].in?(terminal)
      task.sleep(1.0)
    raise IOError("ROCm remote operation timed out")

  def close!(): @http.close!

def main():
  parser = ArgParser(name: "Verify local ROCm package through remote core")
  parser.arg("--url", required: true)
  parser.arg("--token-file", required: true)
  parser.arg("--version", default: "0.1.3")
  parser.arg("--variant", default: "linux-x86_64-gnu-rocm643-prebuilt-source-runtime")
  parser.arg("--source", required: true)
  parser.arg("--output", required: true)
  args = parser.parse_or_raise
  api = CoreClient(args[:url], args[:token_file])
  project = "rocm_upgrade_" + SecureRandom.hex(6)
  route = "/v1/projects/" + project
  created = false
  report = {schema: "torch.registry-rocm-verification.v1", registry: "https://registry.sputnik-lang.online", clean_registry_install: true, project:}
  try:
    source = fs.read_text(args[:source]).replace("from torch.", "from slowpilot.torch.")
    document = {format: "sputnik-notebook", version: 1, title: "ROCm package verification", active_sheet: "main", sheets: [{id: "main", title: "AMD GPU", cells: [{id: "1", kind: "code", source:, mode: "manual"}]}], modules: [], auto_imports: []}
    upload = api.request("PUT", route, {command_id: SecureRandom.hex(16), base_revision: null, document: Json.generate(document), files: {}})
    created = true
    settings = api.request("GET", route + "/worker-settings/main")
    api.request("PUT", route + "/worker-settings/main", {command_id: SecureRandom.hex(16), base_version: settings[:version], runtime_epoch: settings[:runtime_epoch], grants: ["ffi.call", "ffi.load"]})
    before = api.request("GET", route + "/dependencies")
    raise IOError("New project already has packages") unless before[:packages].empty?
    accepted = api.request("POST", route + "/dependencies", {command_id: SecureRandom.hex(16), action: "install_sputnik", source: "slowpilot/torch", version: args[:version], variant: args[:variant], backend: "rocm", bundle: "runtime", allow_native: true, allow_native_large: true})
    installed = api.wait(route + "/dependencies/" + accepted[:id], ["succeeded", "failed", "interrupted"])
    raise IOError("ROCm package install failed: #{Json.generate(installed)}") unless installed[:state] == "succeeded"
    inventory = api.request("GET", route + "/dependencies")
    installed_package = inventory[:packages].find: $it[:name] == "slowpilot/torch"
    raise IOError("Registry Torch variant is absent") unless installed_package and installed_package[:variant] == args[:variant] and installed_package[:version] == args[:version]
    body = {command_id: SecureRandom.hex(16), revision: upload[:revision], sheet_id: "main", cell_id: "1", action: "run"}
    accepted_run = api.request("POST", route + "/runs", body)
    result = api.wait("/v1/runs/" + accepted_run[:run_id], ["succeeded", "failed", "cancelled", "force-stopped", "interrupted"])
    raise IOError("ROCm notebook failed: #{Json.generate(result)}") unless result[:state] == "succeeded"
    value = result[:result][:cells][0][:result]
    raise IOError("AMD training result was not verified") unless value.contains?("PASS ROCm") and value.contains?("forward=14 gradient=[2,4,6]")
    raise IOError("Receipt replay changed RunId") unless api.request("POST", route + "/runs", body) == accepted_run
    report.merge!({verified: true, run_id: accepted_run[:run_id], receipt_replay_same_run: true, "package": installed_package, result: value})
  ensure:
    if created:
      api.request("DELETE", route, {command_id: SecureRandom.hex(16)})
      api.wait(route, ["deleted"])
      report[:project_deleted] = true
    api.close!
  fs.atomic_write_text(args[:output], Json.pretty_generate(report) + "\n")
  print Json.generate(report)
  0
