import supabase from './db-client.js';
export default async function handler(req,res){
  res.setHeader('Access-Control-Allow-Origin','*');
  res.setHeader('Access-Control-Allow-Methods','GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers','Content-Type, Authorization');
  if(req.method==='OPTIONS') return res.status(204).end();
  try{
    if(req.method==='GET'){
      const { experiment_id } = req.query;
      let q=supabase.from('model_runs').select('*').order('created_at',{ascending:false}).limit(200);
      if(experiment_id) q=q.eq('experiment_id', experiment_id);
      const { data, error } = await q;
      if(error) throw error;
      return res.status(200).json(data);
    }
    if(req.method==='POST'){
      const { experiment_id, model_name, model_type, params, metrics, training_config, version } = req.body;
      const { data, error } = await supabase.from('model_runs').insert({ experiment_id, model_name, model_type, params, metrics, training_config, version }).select().single();
      if(error) throw error;
      return res.status(201).json(data);
    }
    if(req.method==='PUT'){
      const { id, ...fields } = req.body;
      const { data, error } = await supabase.from('model_runs').update(fields).eq('id', id).select().single();
      if(error) throw error;
      return res.status(200).json(data);
    }
    if(req.method==='DELETE'){
      const { id } = req.body;
      const { error } = await supabase.from('model_runs').delete().eq('id', id);
      if(error) throw error;
      return res.status(200).json({ok:true});
    }
    return res.status(405).json({error:'Method not allowed'});
  }catch(err){ console.error(err); return res.status(500).json({error:err.message}) }
}
