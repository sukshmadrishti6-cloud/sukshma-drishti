import supabase from './db-client.js';
export default async function handler(req,res){
  res.setHeader('Access-Control-Allow-Origin','*');
  res.setHeader('Access-Control-Allow-Methods','GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers','Content-Type, Authorization');
  if(req.method==='OPTIONS') return res.status(204).end();
  try{
    if(req.method==='GET'){
      const { data, error } = await supabase.from('experiments').select('*').order('created_at',{ascending:false}).limit(100);
      if(error) throw error;
      return res.status(200).json(data);
    }
    if(req.method==='POST'){
      const { title, dataset_name, dataset_hash, qubit_count, feature_map, ansatz, optimizer, shots, status, config } = req.body;
      const { data, error } = await supabase.from('experiments').insert({ title, dataset_name, dataset_hash, qubit_count, feature_map, ansatz, optimizer, shots, status, config }).select().single();
      if(error) throw error;
      return res.status(201).json(data);
    }
    if(req.method==='PUT'){
      const { id, ...fields } = req.body;
      const { data, error } = await supabase.from('experiments').update(fields).eq('id', id).select().single();
      if(error) throw error;
      return res.status(200).json(data);
    }
    if(req.method==='DELETE'){
      const { id } = req.body;
      const { error } = await supabase.from('experiments').delete().eq('id', id);
      if(error) throw error;
      return res.status(200).json({ok:true});
    }
    return res.status(405).json({error:'Method not allowed'});
  }catch(err){ console.error(err); return res.status(500).json({error: err.message}) }
}
