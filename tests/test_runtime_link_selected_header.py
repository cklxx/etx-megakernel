import json
import pytest
from examples.experiment_program.runtime_link import stage_selected_header
from etx.ops.gemv import import_entries
from etx.importer.adapter import emit_host_header


def test_selected_router_descriptor_does_not_reuse_stale_header(tmp_path):
    native=tmp_path/'native';native.mkdir();(native/'imp').mkdir()
    selected=import_entries();selected['dsl_gemv_router_fold']=dict(selected['dsl_gemv_fold'],dsl='router',kernel='dsl_gemv_router_fold',export='etx_tile_dsl_gemv_router_fold')
    old=emit_host_header({k:v for k,v in selected.items() if k!='dsl_gemv_router_fold'})
    (native/'imports.h').write_text(old)
    (native/'imp/imports.json').write_text(json.dumps(selected))
    (native/'vl_plan.txt').write_text('total 160 parts 1 maxlen 2048\n0 dsl_gemv_router_fold 0 router 256 1 1 0 256 128 2048 0\n')
    out=tmp_path/'out';out.mkdir()
    stage_selected_header(native,out)
    header=(out/'imports.h').read_text()
    assert 'etx_imp_dsl_gemv_router_fold_desc' in header
    assert header==emit_host_header(selected)
    assert (native/'imports.h').read_text()==old
    selected.pop('dsl_gemv_router_fold');(native/'imp/imports.json').write_text(json.dumps(selected))
    with pytest.raises(ValueError,match='selected plan export lacks descriptor'):
        stage_selected_header(native,out)


def test_host_rebuild_cannot_overwrite_native_dependencies(tmp_path,monkeypatch):
    from examples.experiment_program import runtime_link as link
    native=tmp_path/'native';native.mkdir();(native/'imp').mkdir()
    (native/'imp/imports.json').write_text(json.dumps(import_entries()))
    (native/'vl_plan.txt').write_text('0 dsl_gemv 0 qkv\n')
    (native/'capture-host.d').write_text('frozen dependency receipt')
    (native/'model.hsaco').write_bytes(b'frozen device')
    (native/'build-identity.json').write_text(json.dumps(dict(hashes={},flags=[],dependency_files={},original_reference_objects={},reused_imports={})))
    host=tmp_path/'host.hip';host.write_text('frozen capture host')
    runtime=tmp_path/'a/b/c/libamdhip64.so';runtime.parent.mkdir(parents=True);runtime.write_bytes(b'library')
    monkeypatch.setattr(link,'verify_files',lambda files:None)
    monkeypatch.setattr(link,'dependencies',lambda out:{})
    monkeypatch.setattr(link,'cpath',str)
    def compile_host(cmd,out,name,**kw):
        (out/(name+'.log')).write_text('')
        (out/(name+'.exit')).write_text('0\n')
        if name=='capture-host-object':
            (out/'capture-host.d').write_text('new selected header receipt')
            assert cmd.index('-I'+str(out))<cmd.index('-I'+str(native))
    monkeypatch.setattr(link,'run',compile_host)
    out=tmp_path/'out'
    link.link_existing(native,out,runtime,host_source=host)
    assert (native/'capture-host.d').read_text()=='frozen dependency receipt'
    assert (out/'capture-host.d').read_text()=='new selected header receipt'
