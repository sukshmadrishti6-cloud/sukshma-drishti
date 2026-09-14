import supabase from './db-client.js';
export default async function handler(req,res){
  res.setHeader('Access-Control-Allow-Origin','*');
  res.setHeader('Access-Control-Allow-Methods','GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers','Content-Type, Authorization');
  if(req.method==='OPTIONS') return res.status(204).end();
  try{
    if(req.method==='GET'){
      const { experiment_id } = req.query;
      let q=supabase.from('inference_logs').select('*').order('created_at',{ascending:false}).limit(100);
      if(experiment_id) q=q.eq('experiment_id', experiment_id);
      const { data, error } = await q;
      if(error) throw error;
      return res.status(200).json(data);
    }
    if(req.method==='POST'){
      const { experiment_id, input_features, prediction, probability, model_used } = req.body;
      const { data, error } = await supabase.from('inference_logs').insert({ experiment_id, input_features, prediction, probability, model_used }).select().single();
      if(error) throw error;
      return res.status(201).json(data);
    }
    return res.status(405).json({error:'Method not allowed'});
  }catch(err){ return res.status(500).json({error:err.message}) }
}
