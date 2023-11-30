import pathlib

import modal
image = modal.Image.debian_slim(python_version="3.9").pip_install("streamlit", "numpy", "pandas","scipy")
stub = modal.Stub(name="Hydropower-App", image=image)
stub.q = modal.Queue.new()
session_timeout = 15 * 60
streamlit_script_local_path = pathlib.Path(__file__).parent / "app.py"
streamlit_script_remote_path = pathlib.Path("/root/app.py")
@stub.function(
    mounts=[
        modal.Mount.from_local_file(
            streamlit_script_local_path,
            remote_path=streamlit_script_remote_path,
        )
    ],
    timeout=session_timeout,
)
def run_streamlit(publish_url: bool = False):
    from streamlit.web.bootstrap import load_config_options, run

    # Run the server. This function will not return until the server is shut down.
    with modal.forward(8501) as tunnel:
        # Reload Streamlit config with information about Modal tunnel address.
        if publish_url:
            stub.q.put(tunnel.url)
        load_config_options(
            {"browser.serverAddress": tunnel.host, "browser.serverPort": 443}
        )
        run(
            main_script_path=str(streamlit_script_remote_path),
            command_line=None,
            args=["--timeout", str(session_timeout)],
            flag_options={},
        )

@stub.function()
@modal.web_endpoint(method="GET")
def share():
    from fastapi.responses import RedirectResponse

    run_streamlit.spawn(publish_url=True)
    url = stub.q.get()
    return RedirectResponse(url, status_code=303)
