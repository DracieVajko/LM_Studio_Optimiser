#!/usr/bin/env python3
"""Clean up DB entries for models not present in LM Studio."""
import asyncio
from lm_optimizer.services.lm_studio import LMStudioClient
from lm_optimizer.database.manager import db_manager

async def main():
    client = LMStudioClient()
    await client.connect(echo_probe=False)
    
    lm_models = await client.list_models()
    lm_model_ids = {m.id for m in lm_models}
    print(f"LM Studio models ({len(lm_models)}):")
    for m in sorted(lm_models, key=lambda x: x.id):
        print(f"  {m.id}")
    
    from lm_optimizer.database.repositories import model_repo
    db_models = model_repo.list_all()
    print(f"\nDB models ({len(model_repo.list_all())}):")
    for m in sorted(model_repo.list_all(), key=lambda x: x.id):
        print(f"  {m.id}")
    
    to_delete = [
        "deepseek-r1-finance-reasoning-14b",
        "fin-o1-14b",
        "google/gemma-4-12b",
        "marco-deepresearch-8b-i1",
        "mistralai/ministral-3-14b-reasoning",
        "qwen3.5-4b",
        "qwen3.8-4b-sft-fable5-glint",
    ]
    
    for mid in to_delete:
        print(f"  DELETING: {mid}")
        
        with db_manager.get_connection() as conn:
            # Get run IDs first
            run_ids = [row[0] for row in conn.execute("SELECT id FROM runs WHERE model_id = ?", (mid,)).fetchall()]
            print(f"  Found {len(run_ids)} runs for {mid}")
            
            if run_ids:
                run_ids_tuple = tuple(run_ids)
                placeholders = ','.join('?' * len(run_ids))
                
                # Delete in correct order (children first)
                # 1. benchmarks
                conn.execute(f"""
                    DELETE FROM benchmarks 
                    WHERE config_id IN (
                        SELECT id FROM configurations 
                        WHERE run_id IN ({','.join('?' * len(run_ids))})
                    )
                """, run_ids)
                
                # quality_results
                conn.execute(f"""
                    DELETE FROM quality_results 
                    WHERE config_id IN (
                        SELECT id FROM configurations 
                        WHERE run_id IN ({','.join('?' * len(run_ids))})
                    )
                """, run_ids)
                
                # presets
                conn.execute(f"DELETE FROM presets WHERE run_id IN ({','.join('?' * len(run_ids))})", run_ids)
                
                # configurations
                conn.execute(f"DELETE FROM configurations WHERE run_id IN ({','.join('?' * len(run_ids))})", run_ids)
                
                # Verify configurations are gone
                remaining = conn.execute(f"""
                    SELECT COUNT(*) FROM configurations 
                    WHERE run_id IN ({','.join('?' * len(run_ids))})
                """, run_ids).fetchone()[0]
                print(f"    Remaining configurations: {remaining}")
            
            # runs
            conn.execute("DELETE FROM runs WHERE model_id = ?", (mid,))
            
            # models
            conn.execute("DELETE FROM models WHERE id = ?", (mid,))
            print(f"  DELETED: {mid}")
        
        print("\nCleanup complete.")
        await client.close()

if __name__ == "__main__":
    import asyncio
    from lm_optimizer.services.lm_studio import LMStudioClient
    from lm_optimizer.database.repositories import model_repo
    from lm_optimizer.database.manager import db_manager
    
    async def main():
        client = LMStudioClient()
        await client.connect(echo_probe=False)
        
        lm_models = await client.list_models()
        lm_model_ids = {m.id for m in lm_models}
        print(f"LM Studio models ({len(lm_models)}):")
        for m in sorted(lm_models, key=lambda x: x.id):
            print(f"  {m.id}")
        
        from lm_optimizer.database.repositories import model_repo
        db_models = model_repo.list_all()
        print(f"\nDB models ({len(model_repo.list_all())}):")
        for m in sorted(model_repo.list_all(), key=lambda x: x.id):
            print(f"  {m.id}")
        
        to_delete = [
            "deepseek-r1-finance-reasoning-14b",
            "fin-o1-14b",
            "google/gemma-4-12b",
            "marco-deepresearch-8b-i1",
            "mistralai/ministral-3-14b-reasoning",
            "qwen3.5-4b",
            "qwen3.8-4b-sft-fable5-glint",
        ]
        
        for mid in to_delete:
            print(f"  DELETING: {mid}")
            
            with db_manager.get_connection() as conn:
                # Get run IDs first
                run_ids = [row[0] for row in conn.execute("SELECT id FROM runs WHERE model_id = ?", (mid,)).fetchall()]
                print(f"  Found {len(run_ids)} runs for {mid}")
                
                if run_ids:
                    # Delete in correct order (children first)
                    # 1. benchmarks
                    conn.execute(f"""
                        DELETE FROM benchmarks 
                        WHERE config_id IN (
                            SELECT id FROM configurations 
                            WHERE run_id IN ({','.join('?' * len(run_ids))})
                        )
                    """, run_ids)
                    
                    # quality_results
                    conn.execute(f"""
                        DELETE FROM quality_results 
                        WHERE config_id IN (
                            SELECT id FROM configurations 
                            WHERE run_id IN ({','.join('?' * len(run_ids))})
                        )
                    """, run_ids)
                    
                    # presets
                    conn.execute(f"DELETE FROM presets WHERE run_id IN ({','.join('?' * len(run_ids))})", run_ids)
                    
                    # configurations
                    conn.execute(f"DELETE FROM configurations WHERE run_id IN ({','.join('?' * len(run_ids))})", run_ids)
                
                # runs
                conn.execute("DELETE FROM runs WHERE model_id = ?", (mid,))
                
                # models
                conn.execute("DELETE FROM models WHERE id = ?", (mid,))
                print(f"  DELETED: {mid}")
        
        print("\nCleanup complete.")
        await client.close()

    asyncio.run(main())

if __name__ == "__main__":
    import asyncio
    from lm_optimizer.services.lm_studio import LMStudioClient
    from lm_optimizer.database.repositories import model_repo
    from lm_optimizer.database.manager import db_manager
    
    async def main():
        client = LMStudioClient()
        await client.connect(echo_probe=False)
        
        lm_models = await client.list_models()
        lm_model_ids = {m.id for m in lm_models}
        print(f"LM Studio models ({len(lm_models)}):")
        for m in sorted(lm_models, key=lambda x: x.id):
            print(f"  {m.id}")
        
        from lm_optimizer.database.repositories import model_repo
        db_models = model_repo.list_all()
        print(f"\nDB models ({len(model_repo.list_all())}):")
        for m in sorted(model_repo.list_all(), key=lambda x: x.id):
            print(f"  {m.id}")
        
        to_delete = [
            "deepseek-r1-finance-reasoning-14b",
            "fin-o1-14b",
            "google/gemma-4-12b",
            "marco-deepresearch-8b-i1",
            "mistralai/ministral-3-14b-reasoning",
            "qwen3.5-4b",
            "qwen3.8-4b-sft-fable5-glint",
        ]
        
        for mid in to_delete:
            print(f"  DELETING: {mid}")
            
            with db_manager.get_connection() as conn:
                # Get run IDs first
                run_ids = [row[0] for row in conn.execute("SELECT id FROM runs WHERE model_id = ?", (mid,)).fetchall()]
                print(f"  Found {len(run_ids)} runs for {mid}")
                
                if run_ids:
                    # Delete in correct order (children first)
                    # 1. benchmarks
                    conn.execute(f"""
                        DELETE FROM benchmarks 
                        WHERE config_id IN (
                            SELECT id FROM configurations 
                            WHERE run_id IN ({','.join('?' * len(run_ids))})
                        )
                    """, run_ids)
                    
                    # quality_results
                    conn.execute(f"""
                        DELETE FROM quality_results 
                        WHERE config_id IN (
                            SELECT id FROM configurations 
                            WHERE run_id IN ({','.join('?' * len(run_ids))})
                        )
                    """, run_ids)
                    
                    # presets
                    conn.execute(f"DELETE FROM presets WHERE run_id IN ({','.join('?' * len(run_ids))})", run_ids)
                    
                    # configurations
                    conn.execute(f"DELETE FROM configurations WHERE run_id IN ({','.join('?' * len(run_ids))})", run_ids)
                
                # runs
                conn.execute("DELETE FROM runs WHERE model_id = ?", (mid,))
                
                # models
                conn.execute("DELETE FROM models WHERE id = ?", (mid,))
                print(f"  DELETED: {mid}")
        
        print("\nCleanup complete.")
        await client.close()

    asyncio.run(main())